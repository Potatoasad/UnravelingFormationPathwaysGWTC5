import re

_DIGIT_WORDS = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
}

_PAIR_WORDS = {
    "00": "ZeroZero",
    "01": "ZeroOne",
    "02": "ZeroTwo",
    "03": "ZeroThree",
    "04": "ZeroFour",
    "05": "ZeroFive",
    "06": "ZeroSix",
    "07": "ZeroSeven",
    "08": "ZeroEight",
    "09": "ZeroNine",
    "10": "Ten",
    "11": "Eleven",
    "12": "Twelve",
    "13": "Thirteen",
    "14": "Fourteen",
    "15": "Fifteen",
    "16": "Sixteen",
    "17": "Seventeen",
    "18": "Eighteen",
    "19": "Nineteen",
    "20": "Twenty",
    "21": "TwentyOne",
    "22": "TwentyTwo",
    "23": "TwentyThree",
    "24": "TwentyFour",
    "25": "TwentyFive",
    "26": "TwentySix",
    "27": "TwentySeven",
    "28": "TwentyEight",
    "29": "TwentyNine",
    "30": "Thirty",
    "31": "ThirtyOne",
    "32": "ThirtyTwo",
    "33": "ThirtyThree",
    "34": "ThirtyFour",
    "35": "ThirtyFive",
    "36": "ThirtySix",
    "37": "ThirtySeven",
    "38": "ThirtyEight",
    "39": "ThirtyNine",
    "40": "Forty",
    "41": "FortyOne",
    "42": "FortyTwo",
    "43": "FortyThree",
    "44": "FortyFour",
    "45": "FortyFive",
    "46": "FortySix",
    "47": "FortySeven",
    "48": "FortyEight",
    "49": "FortyNine",
    "50": "Fifty",
    "51": "FiftyOne",
    "52": "FiftyTwo",
    "53": "FiftyThree",
    "54": "FiftyFour",
    "55": "FiftyFive",
    "56": "FiftySix",
    "57": "FiftySeven",
    "58": "FiftyEight",
    "59": "FiftyNine",
    "60": "Sixty",
    "61": "SixtyOne",
    "62": "SixtyTwo",
    "63": "SixtyThree",
    "64": "SixtyFour",
    "65": "SixtyFive",
    "66": "SixtySix",
    "67": "SixtySeven",
    "68": "SixtyEight",
    "69": "SixtyNine",
    "70": "Seventy",
    "71": "SeventyOne",
    "72": "SeventyTwo",
    "73": "SeventyThree",
    "74": "SeventyFour",
    "75": "SeventyFive",
    "76": "SeventySix",
    "77": "SeventySeven",
    "78": "SeventyEight",
    "79": "SeventyNine",
    "80": "Eighty",
    "81": "EightyOne",
    "82": "EightyTwo",
    "83": "EightyThree",
    "84": "EightyFour",
    "85": "EightyFive",
    "86": "EightySix",
    "87": "EightySeven",
    "88": "EightyEight",
    "89": "EightyNine",
    "90": "Ninety",
    "91": "NinetyOne",
    "92": "NinetyTwo",
    "93": "NinetyThree",
    "94": "NinetyFour",
    "95": "NinetyFive",
    "96": "NinetySix",
    "97": "NinetySeven",
    "98": "NinetyEight",
    "99": "NinetyNine",
}


def _digits_to_words(digits: str) -> str:
    """Convert digits one-by-one: 123 -> OneTwoThree."""
    return "".join(_DIGIT_WORDS[d] for d in digits)


def _gw_digits_to_words(digits: str) -> str:
    """
    Convert GW event digits in pairs.

    Examples
    --------
    231123 -> TwentyThreeElevenTwentyThree
    190521 -> NineteenFiveTwentyOne
    151226 -> FifteenTwelveTwentySix
    """
    out = []

    # Process consecutive pairs from the left.
    i = 0
    while i + 1 < len(digits):
        pair = digits[i : i + 2]
        out.append(_PAIR_WORDS[pair])
        i += 2

    # If odd number of digits, spell the final digit normally.
    if i < len(digits):
        out.append(_DIGIT_WORDS[digits[i]])

    return "".join(out)


def _convert_piece(piece: str) -> str:
    """
    Convert one alphanumeric piece to a LaTeX-safe macro fragment.

    Special rule:
    - Consecutive digits immediately after GW are processed in pairs.
    """
    result = []
    i = 0

    while i < len(piece):
        # Special GW + digits rule
        if piece[i : i + 2] == "GW" and i + 2 < len(piece) and piece[i + 2].isdigit():
            result.append("GW")
            i += 2

            j = i
            while j < len(piece) and piece[j].isdigit():
                j += 1

            result.append(_gw_digits_to_words(piece[i:j]))
            i = j
            continue

        char = piece[i]

        if char.isdigit():
            j = i
            while j < len(piece) and piece[j].isdigit():
                j += 1

            result.append(_digits_to_words(piece[i:j]))
            i = j
            continue

        if char.isalpha():
            result.append(char)

        i += 1

    converted = "".join(result)

    if converted:
        converted = converted[0].upper() + converted[1:]

    return converted


def latex_safe_macro_name(name: str) -> str:
    """
    Convert a string into a LaTeX-safe macro name using letters only.

    Examples
    --------
    Chi1A -> ChiOneA
    GW231123 -> GWTwentyThreeElevenTwentyThree
    mu_chi_1_a -> MuChiOneA
    IndependentFreeTiltDropGW231123Fraction -> IndependentFreeTiltDropGWTwentyThreeElevenTwentyThreeFraction
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")

    pieces = re.split(r"[^A-Za-z0-9]+", name)

    macro_name = "".join(_convert_piece(piece) for piece in pieces if piece)

    macro_name = re.sub(r"[^A-Za-z]", "", macro_name)

    if not macro_name:
        raise ValueError(f"Could not make a LaTeX-safe macro name from {name!r}")

    return macro_name