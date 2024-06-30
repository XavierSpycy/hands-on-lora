#!/bin/bash

curl --request POST \
    --url http://localhost:6006/completion \
    --header 'Content-Type: application/json' \
    --data '{
    "prompt": "system\n你是一位智能助手！\nuser\n你好，请问太阳系有多少行星？\nassistant\n", "n_predict": 128}'