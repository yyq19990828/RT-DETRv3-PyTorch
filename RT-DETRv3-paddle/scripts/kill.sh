ps -ef | grep train.py | awk '{print $2}' | xargs kill -9
kill -9 $(lsof -t /dev/nvidia*)
