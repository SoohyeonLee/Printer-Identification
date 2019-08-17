# Printer-Identification
 
## 서론

**계기**
- 소속 연구실 연구과제

**주제**
- 딥러닝을 활용한 인쇄 기기 판별

**설명**
- 8개의 프린터에 대해 같은 영상을 인쇄한 데이터를 기반으로 어느 인쇄기기에서 인쇄된 것인지 판별 

## 성과

**논문**
- 학회 / 학회지 : 한국정보처리학회 (KIPS) / KIPS TRANSACTIONS ON SOFTWARE AND DATA ENGINEERING (KTSDE)
- 국문 제목 : 전역 및 지역 특징 기반 딥러닝을 이용한 프린터 장치 판별 기술
- 영문 제목 : Printer Identification Methods Using Global and Local Feature-Based Deep Learning
- Accepted : 2018.12.01
- DOI : https://doi.org/10.3745/KTSDE.2019.8.1.37

## 실험 환경

**하드웨어 스팩**
- 그래픽 카드 : NVIDIA Titan XP
- RAM : 16GB

**소프트웨어 버전**
- OS : Windows 10 64bit
- CUDA / cuDNN : 8.0 / 5.1
- Python / Tensorflow-gpu : 3.5 / 1.4.0

## 실험 내용

**모델 구조**

<img src="https://github.com/SoohyeonLee/Printer-Identification/blob/master/resource/model structure.png" width="90%"></img>

**전처리**
1. Local Feature (HPF)
- 학습 모델 초기에 입력 데이터에 대해 HPF 적용

2. Global Feature (GLCM)
- tfrecord 생성 시, GLCM 연산을 통해 (256, 256, 12) 크기의 Feature Map을 생성
- 다음 그림은 3bit에 대한 GLCM  


<img src="https://github.com/SoohyeonLee/Printer-Identification/blob/master/resource/glcm.png" width="90%"></img>

- Feature Map을 같은 방향끼리 합산 (256, 256, 12) -> (256, 256, 4)


<img src="https://github.com/SoohyeonLee/Printer-Identification/blob/master/resource/glcm_sum.png" width="90%"></img>

**사용 데이터**
- Random Crop

<img src="https://github.com/SoohyeonLee/Printer-Identification/blob/master/resource/random crop.png" width="90%"></img>

## 실험 결과
- 지역 특징과 전역 특징 모델의 성능 비교

<img src="https://github.com/SoohyeonLee/Printer-Identification/blob/master/resource/accuracy.png" width="90%"></img>

- 지역 특징 모델의 프린터별 정확도

<img src="https://github.com/SoohyeonLee/Printer-Identification/blob/master/resource/local accuracy per printer.PNG" width="90%"></img>


- 전역 특징 모델의 프린터별 정확도

<img src="https://github.com/SoohyeonLee/Printer-Identification/blob/master/resource/global accuracy per printer.PNG" width="90%"></img>

___

**작성일 : 2019.08.17**
