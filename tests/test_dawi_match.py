import pytest

from badil.crawl.dawi import promising, search_queries
from badil.linking.dawi_match import dose_status, evaluate, pack_status, route_status
from badil.scorer.features import strength_numbers


@pytest.mark.parametrize("ours, theirs, expected", [
    ("ENOXA 8000UI A-xa 80mg Sol Inj 2 Ser/0.8ML", "ENOXA 80MG 0,8MG INJ B/2", "match"),
    ("BIVAL 80 mg/12.5 mg Comp BT 30", "BIVAL 80MG/12.5 COMP B/30", "match"),
    ("CO-IRIVEL Comp Pell 300 MG/12.5 MG BT 30", "CO-IRIVEL 300MG/25MG COMP B/90", "conflict"),
    ("ULDINE Comp 20mg Bt 30", "ULDINE 40MG COMP BT/30", "conflict"),
    ("ENOXAMED 4000 UI ANTI XA/0.4 ML SOL INJ BT 2", "ENOXAMED 2000UI SOL INJ B/2", "conflict"),
    ("ENOXA 4000UI A-xa  Sol Inj 2 Ser/0.4ML", "ENOXA 40MG 0,4 INJ B/2", "unknown"),
])
def test_dose_status(ours, theirs, expected):
    assert dose_status(ours, theirs) == expected


@pytest.mark.parametrize("ours, theirs, expected", [
    ("FELDENE Sol Inj 20mg B/2", "FELDENE 20MG SUPP B/15 PCT", "conflict"),
    ("ZARTEG Comp Pell Sec 80 MG BT 30", "ZARTEG 80MG CP BT90", "match"),
])
def test_route_status(ours, theirs, expected):
    assert route_status(ours, theirs) == expected


@pytest.mark.parametrize("ours, theirs, expected", [
    ("VAINCOR 5 MG Comp Pell Sec  Bt 91", "VAINCOR 5MG BT/28 COMP", "conflict"),
    ("VAINCOR 5 MG Comp Pell Sec  Bt 91", "VAINCOR 5MG COMP B/91", "match"),
])
def test_pack_status(ours, theirs, expected):
    assert pack_status(ours, theirs) == expected


def test_strength_numbers_ignore_packs_and_thousands():
    assert strength_numbers("ENOXAMED 10 000 UI ANTI XA/1ML SOL INJ  BT 2") == [10000.0, 1.0]
    assert strength_numbers("MESONE 50MCG SPRAY NASAL F/120") == [50.0]


def test_brand_must_appear_in_designation():
    assert not promising("VAL Comp Pell 80 MG  BT 30", "URIAGE ISOVALE")
    assert promising("VAL Comp Pell 80 MG  BT 30", "VAL 80MG COMP B/30")


def test_queries_try_other_doses_when_first_is_hidden():
    assert search_queries("ENOXA 8000UI A-xa 80mg Sol Inj 2 Ser/0.8ML") == ["enoxa 8000", "enoxa", "enoxa 80", "enoxa 0.8"]


def test_other_pack_is_not_verified():
    medicine = {"brand_name": "ENOXAMED 4000 UI ANTI XA /0.4 ML SOL INJ BT 6 SERINGUES",
                "active_ingredient": "ENOXAPARINE", "public_price": 37.075}
    item = {"designation": "ENOXAMED 4000UI SOL INJ B/2", "dci": "ENOXAPARINE", "price": "16.450"}
    assert evaluate(medicine, item)["status"] == "other_pack"


def test_wrong_product_is_rejected():
    medicine = {"brand_name": "FELDENE Sol Inj 20mg B/2", "active_ingredient": "PIROXICAM", "public_price": 3.78}
    item = {"designation": "FELDENE 20MG SUPP B/15 PCT", "dci": "PIROXICAM", "price": "10.690"}
    assert evaluate(medicine, item)["status"] is None
