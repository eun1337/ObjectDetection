# ObjectDetection

## VScode 중심 코드 작성중
### At "anaconda prompt" -> ~\miniconda3\envs\test\python.exe <-
### (numpy version trouble)
- google colab 사용 시 직접 작성 모듈 임포트 에러 발생
- NameError: name 'load_classes' is not defined 발생

## pallete 다운로드 필요
## detect.py
- def write() 수정함
  -> c1 = tuple(map(int, x[1:3])), c2 = tuple(map(int, x[3:5]))
  -> 정수형으로 반환해야 오류 안생김

## 완성 후 det foler 생성되고 folder 내에 'det_dog-cycle-car'로 분석된 이미지 생성됨
### 아나콘다에서 실행 문구: python detect.py --images dog-cycle-car.png --det det
