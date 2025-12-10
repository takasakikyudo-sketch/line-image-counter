def calculate_score(symbols):
    score = 0
    total = 0

    for s in symbols:
        if s == "double_circle":
            score += 2
            total += 2
        elif s in ("circle", "circle_slash"):
            score += 1
            total += 2
        elif s == "cross":
            total += 2
        else:
            total += 1

    return score, total
