import pandas as pd
import glob
import os

def merge_csv_files(f1, f2, f3):
    # 각 CSV 파일 읽기
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)

    # 데이터프레임 리스트 생성
    dfs = [df1, df2, df3]

    # 모든 데이터프레임 합치기
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    # 결과를 새 CSV 파일로 저장
    combined_df.to_csv("recipes.csv", index=False)

    print(f"합쳐진 데이터가 recipes.csv에 저장되었습니다.")
    return

# 파일 경로 설정
file1 = "10000recipe.csv"
file2 = "food_safety_recipes.csv"
file3 = "recipe_api_recipes.csv"

merge_csv_files(file1, file2, file3)