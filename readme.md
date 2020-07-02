### 준비과정


- 가상환경 만들기   

```
conda create -n yolov3 python=3.7
conda activate yolov3
```

- 필수 패키지 설치   
```
pip install -r ./requirements.txt
```

- wight file download  
```
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights
```

- 데이터셋 구성을 다운로드킷 받기 (옵션)  
```
git clone https://github.com/pythonlessons/OIDv4_ToolKit
```


