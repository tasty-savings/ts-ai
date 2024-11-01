import pandas as pd

# 각 cooking_order 리스트의 항목들을 수정하는 함수
def modify_sentences(orders):
    modified = []
    for order in orders:
        # 이미 "다."로 끝나는 경우
        if order.endswith('다.'):
            modified.append(order)
            continue

        # "다"로 끝나는 경우
        if order.endswith('다'):
            modified.append(order + '.')
            continue

        # 다른 문장부호로 끝나는 경우 (예: ., !, ?)
        if order[-1] in '.!?':
            modified.append(order[:-1] + '다.')
        else:
            # 문장부호 없이 끝나는 경우
            modified.append(order + '다.')

    return modified

def check_brackets_quotes(text):
    # 문자열이 아닌 경우 (예: NaN) 처리
    if not isinstance(text, str):
        return []

    # 리스트 형태의 문자열인 경우 실제 리스트로 변환
    try:
        if text.startswith('[') and text.endswith(']'):
            items = literal_eval(text)
            if isinstance(items, list):
                text = ' '.join(items)
    except:
        pass  # 변환 실패 시 원본 텍스트 사용

    # 검사할 괄호/따옴표 쌍 정의
    brackets = {
        '(': ')',
        '[': ']',
        '{': '}',
        '"': '"',
        "'": "'"
    }

    stack = []
    errors = []

    for i, char in enumerate(text):
        # 여는 괄호/따옴표인 경우
        if char in brackets:
            stack.append((char, i))

        # 닫는 괄호/따옴표인 경우
        elif char in brackets.values():
            if not stack:  # 스택이 비어있으면 닫는 괄호/따옴표가 먼저 나온 것
                errors.append(f"위치 {i}에서 매칭되지 않은 닫는 문자 '{char}' 발견")
                continue

            last_open, start_pos = stack.pop()
            if char != brackets[last_open]:  # 매칭되지 않는 쌍
                errors.append(f"위치 {start_pos}의 '{last_open}'와 위치 {i}의 '{char}'가 매칭되지 않음")

    # 스택에 남아있는 여는 괄호/따옴표 처리
    while stack:
        char, pos = stack.pop()
        errors.append(f"위치 {pos}의 '{char}'에 대한 닫는 문자가 없음")

    return errors


# CSV 파일 읽기
df = pd.read_csv('jori_recipe.csv')
# cooking_order가 문자열이 아닌 리스트 형태라면 먼저 literal_eval로 변환
from ast import literal_eval
df['cooking_order'] = df['cooking_order'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

# 모든 행의 cooking_order 수정
df['cooking_order'] = df['cooking_order'].apply(modify_sentences)

# 모든 열에 대해 검사
for column in df.columns:
    print(f"\n{column} 열 검사 중...")

    for idx, value in df[column].items():
        errors = check_brackets_quotes(str(value))
        if errors:
            print(f"\n행 {idx}의 오류:")
            print(f"값: {value}")
            for error in errors:
                print(error)

print("\n검사 완료!")

# 수정된 데이터프레임을 CSV 파일로 저장
df.to_csv('jori_recipe.csv', index=False)