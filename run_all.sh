#!/usr/bin/env bash
/usr/bin/python3.6 gru_sl_50.py
echo "[FINISHED] GRU Sequence Length 50"
/usr/bin/python3.6 lstm_sl_24.py
echo "[FINISHED] LSTM Sequence Length 24"
/usr/bin/python3.6 lstm_sl_50.py
echo "[FINISHED] LSTM Sequence Length 50"