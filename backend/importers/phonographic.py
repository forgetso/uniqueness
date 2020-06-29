import csv
from backend.db.db import connect, execute_values_wrap
from backend.lib.setup import db_setup
from backend.settings import PHONOGRAPHIC_FILE


def main():
    target_file = PHONOGRAPHIC_FILE
    process_data(target_file)


def process_data(target_file):
    with open(target_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        # headers
        # http://www.iphod.com/details/
        # Indx	Word	UnTrn	StTrn	NSyll	NPhon	unsDENS	unsFDEN	unsLDEN	unsCDEN	strDENS	strFDEN	strLDEN	strCDEN	unsBPAV	unsFBPAV	unsLBPAV	unsCBPAV	strBPAV	strFBPAV	strLBPAV	strCBPAV	unsTPAV	unsFTPAV	unsLTPAV	unsCTPAV	strTPAV	strFTPAV	strLTPAV	strCTPAV	unsPOSPAV	unsFPOSPAV	unsLPOSPAV	unsCPOSPAV	strPOSPAV	strFPOSPAV	strLPOSPAV	strCPOSPAV	unsLCPOSPAV	unsFLCPOSPAV	unsLLCPOSPAV	unsCLCPOSPAV	strLCPOSPAV	strFLCPOSPAV	strLLCPOSPAV	strCLCPOSPAV	SFreq	SCDcnt
        records = []
        for row in reader:
            record = {'word': row['Word'].lower()}
            other = {k: v for k, v in row.items() if k not in ('Word', 'Indx', '')}
            record.update(other)
            records.append(record)
    db_setup(phonographic=True)
    save_phonographic_records(records)


def save_phonographic_records(records):
    fieldnametuple = tuple(records[0].keys())
    execute_values_wrap('phonographic', fieldnametuple, [list(x.values()) for x in records])


def setup():
    main()


if __name__ == "__main__":
    setup()
