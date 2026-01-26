@echo off
chcp 65001 > nul
echo ======================================================================
echo   Seed 탐색: 여러 seed로 모델 훈련
echo ======================================================================
echo.

set SEEDS=42 123 789 2026 456

for %%s in (%SEEDS%) do (
    echo.
    echo ======================================================================
    echo   Seed %%s 실행 중...
    echo ======================================================================
    
    REM main.py의 RANDOM_SEED 변경
    powershell -Command "(Get-Content src\main.py) -replace 'RANDOM_SEED = \d+', 'RANDOM_SEED = %%s' | Set-Content src\main.py"
    
    REM 실행
    call conda activate env
    python src\main.py
    
    REM submission 백업
    if exist outputs\submission.json (
        copy outputs\submission.json outputs\submission_seed%%s.json > nul
        echo ✓ Submission 저장: outputs\submission_seed%%s.json
    )
)

echo.
echo ======================================================================
echo   완료! outputs 폴더에서 각 seed의 결과를 확인하세요
echo ======================================================================
echo.
echo outputs\submission_seed*.json 파일들을 AI Factory에 제출해서
echo 어느 seed가 가장 좋은 성능인지 확인하세요!

REM 원래 seed로 복원
powershell -Command "(Get-Content src\main.py) -replace 'RANDOM_SEED = \d+', 'RANDOM_SEED = 42' | Set-Content src\main.py"
