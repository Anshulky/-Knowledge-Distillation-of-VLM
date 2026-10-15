"""Regression tests for the scorer (ie643.eval). Each case is a bug that changed reported results.

  .venv/bin/python -m unittest tests.test_eval_match -v
"""

from __future__ import annotations

import unittest

from ie643.eval.gt_clean import clean_gold, is_artifact
from ie643.eval.match import list_prf, parse_number, values_match
from ie643.eval.score import _prf, score_record


class NumberParsing(unittest.TestCase):
    def test_separators_follow_one_rule(self) -> None:
        cases = {
            "54,000": 54000,  # was 54.0 while "54.000" was 54000
            "54.000": 54000,
            "17,727": 17727,
            "1.234,56": 1234.56,
            "1,234.56": 1234.56,
            "1,000,000": 1_000_000,
            "48,88 EUR": 48.88,
            "0.055": 0.055,  # leading zero: decimal, not thousands
            "Rp. 54.000": 54000,
            "RM 59.20": 59.20,
            "TOTAL : 623.21 EUR": 623.21,
            "5,5%": 5.5,
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(parse_number(text), want)

    def test_long_numbers_are_one_token(self) -> None:
        # the old regex split "110000" into "110" + "000" and kept "000"
        self.assertEqual(parse_number("*110000"), 110000)


class AmountMatching(unittest.TestCase):
    def test_different_amounts_do_not_match(self) -> None:
        # all of these passed through the fuzzy-string fallback (ANLS >= 0.5) or the 1% tolerance
        for field, pred, gold in [
            ("subtotal", "17,121", "17,727"),
            ("tax", "1,737", "1,773"),
            ("change", "160,000", "80,500"),
            ("total", "18,858", "18,999"),
            ("total", "388,000", "194,000"),
        ]:
            with self.subTest(pred=pred, gold=gold):
                self.assertFalse(values_match(field, pred, gold))

    def test_same_amount_in_other_formats_matches(self) -> None:
        for field, pred, gold in [
            ("total", "54,000", "54.000"),
            ("total", "48,88 EUR", 48.88),
            ("total", "691.96 USD", "TOTAL : 691.96 USD"),
            ("total", "$8.90", "8.90"),
        ]:
            with self.subTest(pred=pred, gold=gold):
                self.assertTrue(values_match(field, pred, gold))

    def test_discount_ignores_sign_convention(self) -> None:
        self.assertTrue(values_match("discount", "- 6.11", "DISCOUNT(1.03%): (-)  6.11"))
        self.assertFalse(values_match("discount", "- 6.20", "DISCOUNT(1.03%): (-)  6.11"))

    def test_text_amount_falls_back_to_string_match(self) -> None:
        self.assertTrue(values_match("total", "FREE", "free"))


class ListRows(unittest.TestCase):
    def test_label_only_tax_row_matches_itself(self) -> None:
        # _strip_labels used to turn "GST" into "", so the row could never match
        row = {"label": "GST", "rate": "6%", "amount": "1.46"}
        self.assertEqual(list_prf("taxes", [row], [row]), (1, 0, 0))

    def test_line_item_amount_is_numeric(self) -> None:
        gold = [{"name": "SAMGYOPSAL", "quantity": "2", "unit_price": None, "amount": "194,000"}]
        pred = [{"name": "SAMGYOPSAL", "quantity": "2", "unit_price": "194,000", "amount": "388,000"}]
        self.assertEqual(list_prf("line_items", pred, gold), (0, 1, 1))


class GroundTruthCleaning(unittest.TestCase):
    def test_placeholders_become_null(self) -> None:
        g = clean_gold({"invoice_number": "[]", "tax": "-", "total": "10.00"}, "fatura2")
        self.assertIsNone(g["invoice_number"])
        self.assertIsNone(g["tax"])
        self.assertEqual(g["total"], "10.00")

    def test_fatura_payment_method_not_scored(self) -> None:
        g = clean_gold({"payment_method": "Bank Name   State Bank\nBranch Name  X"}, "fatura2")
        self.assertIsNone(g["payment_method"])
        self.assertEqual(clean_gold({"payment_method": "cash"}, "cord-v2")["payment_method"], "cash")

    def test_dict_text_is_unpacked(self) -> None:
        gold = {
            "total": "{'total': '40,000', 'cash': '50,000', 'change': '10,000'}",
            "subtotal": "{'subtotal': 'Rp 36.000', 'tax': 'Rp 3.600'}",
        }
        g = clean_gold(gold, "invoices-and-receipts-v2")
        self.assertEqual(g["total"], "40,000")
        self.assertEqual(g["subtotal"], "Rp 36.000")
        self.assertTrue(values_match("total", "40,000", g["total"]))
        self.assertTrue(is_artifact("invoices-and-receipts-v2", "total", gold["total"]))

    def test_repeated_amount_is_one_value(self) -> None:
        g = clean_gold({"total": "$49.70 49.70", "tax": "€2,37 €1,05", "vendor": "UNO UNO"}, "invoices-and-receipts-v2")
        self.assertEqual(g["total"], "$49.70")
        self.assertTrue(values_match("total", "$49.70", g["total"]))
        self.assertEqual(g["tax"], "€2,37 €1,05")  # two different amounts stay as printed
        self.assertEqual(g["vendor"], "UNO UNO")  # no digits: untouched

    def test_clean_does_not_mutate_input(self) -> None:
        gold = {"invoice_number": "[]"}
        clean_gold(gold, "fatura2")
        self.assertEqual(gold["invoice_number"], "[]")


class Scores(unittest.TestCase):
    def test_f1_is_not_rounded(self) -> None:
        # 0.98248 was stored as 0.9825 and printed as "0.983"
        f1 = _prf(39640, 652, 762)["f1"]
        self.assertAlmostEqual(f1, 2 * 39640 / (2 * 39640 + 652 + 762), places=12)

    def test_identical_prediction_scores_perfectly(self) -> None:
        gold = {
            "total": "17,500",
            "subtotal": "15,909",
            "taxes": [{"label": "VAT", "rate": "10%", "amount": "1,591"}],
            "line_items": [{"name": "Choco Devil", "quantity": "1", "unit_price": None, "amount": "15,909"}],
        }
        for field, (tp, fp, fn) in score_record(gold, gold).items():
            with self.subTest(field=field):
                self.assertEqual((fp, fn), (0, 0))


if __name__ == "__main__":
    unittest.main()
