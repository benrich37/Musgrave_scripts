

header_line = '*************** JDFTx 1.7.0 (git hash 5f12716a) ***************'

def clean_out(filepath):
    out_clean = []
    with open(filepath) as file:
        for line in file:
            if header_line in line:
                out_clean = []
            out_clean.append(line.rstrip('\n'))
    return out_clean


def get_fftbox(out_clean):
    for line in out_clean:
        if 'Chosen fftbox' in line:
            use = line
            break
    clean_line = list(filter(lambda item: len(item) != 0, use.split(' ')))
    trimmed = clean_line[clean_line.index('[') + 1:clean_line.index(']')]
    fftbox = [eval(i) for i in trimmed]
    return fftbox