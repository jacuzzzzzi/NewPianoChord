import os
import argparse
from basic_pitch.inference import predict_and_save, Model
from basic_pitch import ICASSP_2022_MODEL_PATH

basic_pitch_model = Model(ICASSP_2022_MODEL_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MP3 to Midi, Please write the name of audio file")
    parser.add_argument("--input_audio", type=str, required=True,
                        help="입력할 멜로디 MIDI 파일 경로")
    arg = parser.parse_args()

    # 변환 실행
    predict_and_save([arg.input_audio], output_directory=os.getcwd(), save_midi=True,
                     sonify_midi=False,
                     save_model_outputs=False,
                     save_notes=False,
                     model_or_model_path=basic_pitch_model)

    print(f"🎵 변환 완료: {arg.output_midi}")
