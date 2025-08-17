import numpy as np
import sounddevice as sd
import soundfile as sf
import time
from datetime import datetime
import csv

"""sounddeviceの初期設定"""
fs = 44100
sd.default.samplerate = fs #サンプリングレート
channels = [8, 1] 
sd.default.channels = channels # [マイク, スピーカー]
dev = [13,13] # [Input, Output] (ASIO)
sd.default.device = dev
sd.default.dtype = 'float64'
sd.default.latency = 'low' # 遅延少なめ

RATE = fs
READWAV_soundfile = 'Chirp_ex.wav' # 読み込む音源ファイル
READWAV_sounddata, READWAV_samplerate = sf.read(READWAV_soundfile)

##### Single Subject #####
#SUBJECT_NAME="Shibuya"
#ACTIVE_WAV_FOLDER = "./data"
#WRITE_CSV_FILE = "./TrainSet.csv"

### Cross Subject #####
SUBJECT_NAME="Shibuya"
ACTIVE_WAV_FOLDER = "./data/sub"
WRITE_CSV_FILE = "./csv/TrainSet.csv"

CHANNEL = 8   # マイクの数
INF=100000000 
LABEL = -INF
RECORD_NUM = 20   # データを取る回数


if __name__ == '__main__':

    cnt = 0
    print("学習するデータラベルを入力 (0 ~ 11)")
    t = int(input())
    LABEL = t

    print("CLASS_LABEL:", LABEL)
    time.sleep(2)

    for _ in range(0, RECORD_NUM):
        print("RECORD_NUM:", cnt)
        # 再生・録音を同時に行うメソッド
        myrecording = sd.playrec(READWAV_sounddata, READWAV_samplerate)            
        sd.wait() # playrec()の処理が終わるまでまつ

        files = []
        for ch_index in range(0, CHANNEL):
            filename = ACTIVE_WAV_FOLDER+"/"+datetime.today().strftime("%Y%m%d_%H%M%S")+"_"+str(cnt)+"_C"+str(ch_index)+"_USR"+SUBJECT_NAME+".wav"
            files.append(filename)
            # 音データをwavで保存
            sf.write(filename, myrecording[:,ch_index], READWAV_samplerate)

        #print("wav file saved")

        with open(WRITE_CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            written_data = [LABEL,-1]
            for filen in files:
                written_data.append(filen)
            writer.writerows([written_data])
            #print("write csv file")

        cnt += 1
        if cnt == RECORD_NUM + 1:
            break

        time.sleep(3)

    print("サンプリング終了")

