from dataclasses import dataclass


@dataclass
class NIDFrontData:
    name: str | None = None
    father_name: str | None = None
    mother_name: str | None = None
    spouse_name: str | None = None
    date_of_birth: str | None = None
    nid_number: str | None = None


@dataclass
class NIDBackData:
    address: str | None = None
    blood_group: str | None = None
    issue_date: str | None = None
    place_of_birth: str | None = None
