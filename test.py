from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import torch
from accelerate import Accelerator
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import random
from datasets import Dataset
from datetime import datetime

import llama.token_limit    # 自作関数
import create_dataset_Kaggle  #自作

########
# モデルとトークナイザーのロード
########
# QLoRA?
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Acceleratorの初期化
accelerator = Accelerator()

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # ここはLlama2やLlama3のモデル名を指定
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 新しいパディングトークンを追加（これが無いとパディングトークンを設定できずエラーが発生する）
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config = bnb_config, device_map="auto")

# GPUの割当確認↓
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")


########
# LoRA設定
########
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)





########
# PEFTモデルの取得
########
peft_model = get_peft_model(model, lora_config)

pipe = pipeline("text-generation", model=peft_model, tokenizer=tokenizer,max_new_tokens=100)
"""
def ask(pipe,text):
    out=pipe(text)
    out=out[0]["generated_text"][len(text):]
    return out

system_prompt="You are a professional chemist. Predict the melting point of the following compound."
text = system_prompt

ask(pipe,text)
"""




########
# データセットのロード
########
# df = pd.read_csv("sample_Kaggle_train_100.csv")
df = pd.read_csv("../KaggleFakeNewsDetectionDataset/train.csv")
# df = create_dataset_Kaggle.main("../KaggleFakeNewsDetectionDataset/test.csv", "../KaggleFakeNewsDetectionDataset/submit.csv")

# 前処理
df = llama.token_limit.main(df.copy(), text_column='text', token_limit=2048, action='remove')
df["label"] = df["label"].replace({0: "Real", 1:"fake"})
print("----------------------------------")
print(df)
print("----------------------------------")

# df=pd.read_csv("231228best_reason_record.csv")

dataset=df.to_dict(orient="records")
#######
# orient="records"の挙動例↓
"""
[
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25}
]
"""
#######



##########
# 渡されたデータを元に、特定のフォーマットで文字列を生成
##########
def gen_compound_text(article_record, prefix="Example"):
    title = article_record["title"]
    text = article_record["text"]
    label = article_record["label"]
    prompt=f"""
    #{prefix} Data
    ##News title: {title}
    ##News text: {text} 
    ##label : {label}
    """
    return prompt


##########
# 与えられたデータセットから指定された数のトレーニングプロンプトと1つのテストプロンプトを生成するために使用
##########
def generate_question_prompt(dataset,test_id,n_prompt_examples=5):
    train_ids=[i for i in range(len(dataset))]
    train_ids.remove(test_id)
    prompt=""

    #train prompt
    for _ in range(n_prompt_examples):
        id=random.choice(train_ids)
        prompt+=gen_compound_text(dataset[id],
                                reason=dataset[id]["Reason"],
                                prediction=dataset[id]["Prediction(integer)"])
        prompt+="\n"

    #test prompt
    prompt+=gen_compound_text(dataset[test_id],prefix="Test")
    #    prompt+="""
    ##Output: Reason, Prediction
    #    """

    return prompt

# 与えられたテキストリストをシャッフルし、トークナイザーを使用してトークン化したデータセットを生成
def prepare_dataset(context_list, tokenizer):
    data_list = [{"text": i} for i in context_list]
    random.shuffle(data_list)

    # tokenize
    dataset = Dataset.from_dict(
        {"text": [item["text"] for item in data_list[:]]})
    
    dataset = dataset.map(lambda samples: tokenizer(samples['text']), batched=True)

    return dataset


#とりあえず初めの10件をテストデータにする
n_test=10

train_text_list=[]
for id in range(len(dataset)):
    prompt=gen_compound_text(dataset[id])
    train_text_list.append(prompt)
tokenized_dataset = prepare_dataset(train_text_list[n_test:], tokenizer)
tokenized_dataset_test = prepare_dataset(train_text_list[:n_test], tokenizer)

print(tokenized_dataset)                # データ構造確認
print(tokenized_dataset[0]["text"])       # 生データ一個表示


########
# トレーニングの設定
########
tokenizer.pad_token = tokenizer.eos_token
train_args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        num_train_epochs=5,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=100,
        save_total_limit=1,
        output_dir='outputs/'+datetime.now().strftime('%Y%m%d%H%M%S'),
    )

trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=tokenized_dataset,
    args=train_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

training_result = trainer.train()
print("Training Loss:", training_result.training_loss)

peft_model.save_pretrained("./output_llama3_peft")  # ここに保存するパスを指定
tokenizer.save_pretrained("./output_llama3_peft")    # トークナイザーの保存も忘れずに

"""
epoch = 1
ans_dict={}
for i in range(epoch):
    training_result=trainer.train()
    res=ask(pipe,text)
    ans_dict[i]={"out":res,"loss":training_result.training_loss    }
    print(res)
"""
