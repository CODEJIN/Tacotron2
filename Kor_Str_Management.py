import re
from hanja import hangul

#Only until 100
def Count_Number(n):
    count_to_korean = [""] + ["한","두","세","네","다섯","여섯","일곱","여덟","아홉"]
    count_tenth_dict = {
            "한십": "열",
            "두십": "스물",
            "세십": "서른",
            "네십": "마흔",
            "다섯십": "쉰",
            "여섯십": "예순",
            "일곱십": "일흔",
            "여덟십": "여든",
            "아홉십": "아흔",
    }
    if n < 10:
        return count_to_korean[n];
    elif n >= 100:
        return Read_Number(n);

    tenth, digit = divmod(n, 10);
    korean_Count = count_to_korean[tenth] + "십" + count_to_korean[digit];

    for original_Char in count_tenth_dict.keys():
        korean_Count = korean_Count.replace(original_Char, count_tenth_dict[original_Char]);

    return korean_Count;

# Referred: http://soooprmx.tistory.com/entry/파이썬-숫자를-한글로-읽는-함수
def Read_Number(n, rate_Level = 0):
    if n == 0 and rate_Level == 0:
        return "영";

    rate_Unit_Suffix_List = ["", "만", "억", "조", "경", "해"];
    rated_Prefix = ""
    higher_Rate_Number, main_Number = divmod(n, 10000)
    if higher_Rate_Number > 0:
        rated_Prefix = Read_Number(higher_Rate_Number, rate_Level = rate_Level + 1);

    unit_List = ['', '십', '백', '천'];
    number_List = list('일이삼사오육칠팔구');
    result_List = []
    unit_Index = 0
    while main_Number > 0:
        main_Number, remain = divmod(main_Number, 10)
        if remain > 0:
            result_List.append(number_List[remain-1] + unit_List[unit_Index])
        unit_Index += 1
    result_List = list(reversed(result_List));

    korean_Number = rated_Prefix + "".join(result_List) + rate_Unit_Suffix_List[rate_Level];

    if korean_Number[0] == "일" and len(korean_Number) > 1:
        korean_Number = korean_Number[1:];

    return korean_Number;

def Number_to_String(string):
    replace_Dict = {
        "0": "공",
        "1": "일",
        "2": "이",
        "3": "삼",
        "4": "사",
        "5": "오",
        "6": "육",
        "7": "칠",
        "8": "팔",
        "9": "구",
    }

    if type(string) == int:
        string = str(string);

    new_String = string;
    for original_Char in replace_Dict.keys():
        new_String = new_String.replace(original_Char, replace_Dict[original_Char]);

    return new_String;

def String_to_Token_List(string):
    #English or special char....
    if re.search(r'[가-힣\d\s\.,\?!]+', string) is None or re.search(r'[가-힣\d\s\.,\?!]+', string).group() != string:
        print(string);
        return False;

    regex_DtoS = r'(?:^|[^\d])(\d{4})(?:$|[^\d])';
    string = re.sub(
        regex_DtoS,
        lambda x: re.sub(r'\d{4}', lambda y: Number_to_String(y.group()), x.group()),
        string
        )

    regex_CtoS1 = r"([+-]?\d[\d,]*)[\.]?\d*"
    regex_CtoS2 = r"(시|명|가지|살|마리|포기|송이|수|톨|통|점|개|벌|척|채|다발|그루|자루|줄|켤레|그릇|잔|마디|상자|사람|곡|병|판)"
    string = re.sub(
        regex_CtoS1 + regex_CtoS2,
        lambda x: re.sub(regex_CtoS1, lambda y: Count_Number(int(y.group())), x.group()),
        string);

    regex_NtoS = r"([+-]?\d[\d,]*)[\.]?\d*"
    string = re.sub(
        regex_NtoS,
        lambda x: Read_Number(int(x.group())),
        string);

    token_List = [];
    token_List.append(0); #<EOS>
    for char in string:
        if char == " ":
            token_List.append(2);
            continue;
        elif char == ".":
            token_List.append(71);
            continue;
        elif char == ",":
            token_List.append(72);
            continue;
        elif char == "?":
            token_List.append(73);
            continue;
        elif char == "!":
            token_List.append(74);
            continue;
        elif hangul.is_hangul(char):
            onset, nucleus, coda = hangul.separate(char);
            onset += 3;
            nucleus += 3 + 19;
            coda += 3 + 19 + 21;
            token_List.extend([onset, nucleus, coda]);
        else:
            raise Exception("Not handled letter");

    token_List.append(1); #<EOE>

    return token_List;
