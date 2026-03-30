# 👁️ CORE - SOTA Vision Pipeline & Agent

> **CORE 시스템**의 AI 분석 엔진입니다. 악천후 위성 이미지 복원, RS-Mamba 기반 지형 변화 탐지, Amodal 분할 및 LangChain 기반 시각 정보 요약 리포팅을 수행합니다.

## 🛠 Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Vision Models:** DDGAN (Restoration), RS-Mamba (Backbone), Amodal Segmentation
* **Agent:** LangChain, OpenAI API (or Local LLM)
* **Data Handling:** GeoPandas, OpenCV

## 📂 Directory Structure
* `data/`: 데이터셋 폴더 (xBD 등) - **⚠️ `.gitignore`에 등록되어 GitHub에 업로드되지 않습니다.**
* `models/`: DDGAN, RS-Mamba 등 딥러닝 모델 아키텍처 및 체크포인트 로드 코드
* `utils/`: 데이터 로더(DataLoader), Tiling, 노이즈 증강(Augmentation) 스크립트
* `agent/`: Segmentation 마스크를 자연어로 변환하는 LangChain 로직
* `inference.py`: 단일/스트림 이미지 통합 추론 파이프라인

## 🚀 Getting Started
1. **환경 설정:**
   ```bash
   git clone [https://github.com/](https://github.com/)[Organization-Name]/core-vision.git
   cd core-vision
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2.  **데이터셋 준비:**
    xBD 데이터셋을 다운로드하여 `data/` 디렉토리에 위치시킵니다. (상세 구조는 `utils/dataloader.py` 주석 참고)
3.  **추론(Inference) 테스트:**
    ```bash
    python inference.py --input_dir data/sample_images --output_dir results/
    ```

## ⚠️ Data & Weights License Notice

본 코드 베이스는 MIT License를 따르나, 본 프로젝트에서 사용되는 **xBD 데이터셋** 및 특정 사전 학습된 모델 가중치(Pre-trained Weights)는 원저작자의 비상업적/연구용 라이선스를 따릅니다. 데이터셋은 절대 본 레포지토리에 커밋하지 마십시오.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
