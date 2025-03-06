import miditok
import miditoolkit
import random
import argparse
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import miditok
import miditoolkit


def midi_to_tokens(midi_path, mask_ratio=0.2):
    """
    _summary_

    Args:
        midi_path (str): _description_
        mask_ratio (float, optional): _description_. Defaults to 0.2.

    Returns:
        token: _description_
    """
    tokenizer = miditok.REMI()  # 다양한 변환 방식 가능 (MIDILike, TSD 등)

    # MIDI 파일 로드
    midi_data = miditoolkit.MidiFile(midi_path)

    # MIDI → 토큰 변환
    tokens = tokenizer(midi_data)

    # 마스킹할 토큰 선택
    num_masks = int(len(tokens) * mask_ratio)
    masked_tokens = tokens.copy()
    mask_indices = random.sample(range(len(tokens)), num_masks)

    for idx in mask_indices:
        masked_tokens[idx] = "[MASK]"  # 특정 토큰을 마스킹

    token_str = " ".join(masked_tokens)  # LLaMA가 이해할 수 있도록 텍스트 형태로 변환

    return token_str, tokens  # (마스킹된 입력, 원본 정답)


def prepare_dataset(midi_files, tokenizer: LlamaTokenizer):
    """_summary_

    Args:
        midi_files (_type_): _description_
        tokenizer (LlamaTokenizer): _description_

    Returns:
        _type_: _description_
    """
    masked_inputs = []
    labels = []

    for midi_path in midi_files:
        masked, original = midi_to_tokens(midi_path)
        masked_inputs.append(masked)
        labels.append(original)

    dataset = Dataset.from_dict({
        "input_ids": [tokenizer(m, return_tensors="pt").input_ids for m in masked_inputs],
        "labels": [tokenizer(l, return_tensors="pt").input_ids for l in labels],
    })

    return dataset


def generate_accompaniment(midi_path):
    """_summary_

    Args:
        midi_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    masked_tokens, _ = midi_to_tokens(midi_path)
    input_ids = tokenizer(masked_tokens, return_tensors="pt").input_ids

    with torch.no_grad():
        output = model.generate(input_ids, max_length=512)

    generated_tokens = tokenizer.decode(output[0])
    return generated_tokens


def tokens_to_midi(tokens, output_path):
    """_summary_

    Args:
        tokens (_type_): _description_
        output_path (_type_): _description_
    """
    tokenizer = miditok.REMI()
    generated_midi = tokenizer.tokens_to_midi(tokens)
    generated_midi.dump(output_path)
    print(f"🎵 반주 MIDI 파일이 저장되었습니다: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MusicBERT 기반 반주 생성기")

    parser.add_argument("--input_midi", type=str, required=True,
                        help="입력할 멜로디 MIDI 파일 경로")
    parser.add_argument("--output_midi", type=str, default="output.mid",
                        help="생성된 반주를 저장할 MIDI 파일 경로")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="사용할 MusicBERT 모델 (기본값: facebook/musicbert)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="생성할 반주의 최대 길이 (기본값: 512 토큰)")
    args = parser.parse_args()
    # LLaMA 모델 및 토크나이저 로드
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    model = LlamaForCausalLM.from_pretrained(args.model)
    dataset = prepare_dataset(*args.input_midi)

    # Training
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
