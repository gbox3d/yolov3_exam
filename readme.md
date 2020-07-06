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

- weight file download  
```
# yolov3
wget -P model_data https://pjreddie.com/media/files/yolov3.weights

# yolov3-tiny
wget -P model_data https://pjreddie.com/media/files/yolov3-tiny.weights
```

### 데이터셋 구성

- 다운로드킷 받기 (옵션)  
```
git clone https://github.com/pythonlessons/OIDv4_ToolKit
```

- 데이터 다운로드  

OIDv4_ToolKit/ 이 있는 위치에서 다음 명령을 실행한다.  
```sh
python OIDv4_ToolKit/main.py downloader --classes Bird Person --type_csv train --limit 2000
python OIDv4_ToolKit/main.py downloader --classes Bird Person --type_csv test --limit 200
```
다운로드 위치는 기본적으로 OID/Dataset 이다. 위치를 바꾸고 싶은면 OIDv4_ToolKit/main.py 의 DEFAULT_OID_DIR 변수를 수정한다.   

### 참고 자료

