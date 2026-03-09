import re

_word_values = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
    'hundred': 100, 'thousand': 1000, 'lakh': 100000, 'crore': 10000000,
    'lakhs': 100000, 'crores': 10000000
}

def words_to_digits_string(text: str) -> list[str]:
    """
    Parses a string that may contain numeric words like 'Fifty Lakh' 
    and returns a list of detected numbers as digit strings.
    """
    words = re.findall(r'[a-z]+', text.lower())
    
    current_val = 0
    final_val = 0
    found_any = False
    
    results = []
    
    for w in words:
        if w in _word_values:
            found_any = True
            val = _word_values[w]
            if val == 100:
                if current_val == 0: current_val = 1
                current_val *= val
            elif val >= 1000:
                if current_val == 0: current_val = 1
                current_val *= val
                final_val += current_val
                current_val = 0
            else:
                current_val += val
        else:
            if found_any:
                total = final_val + current_val
                if total > 0:
                    results.append(str(total))
                final_val = 0
                current_val = 0
                found_any = False
                
    if found_any:
        total = final_val + current_val
        if total > 0:
            results.append(str(total))
            
    return results
