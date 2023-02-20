import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--logname',
                    help='filename of log file to extract data from',
                    type=str)
parser.add_argument('-o', '--newname',
                    help='what your new files will be called',
                    type=str,
                    default=None)
parser.add_argument('-m', '--mem',
                    help='mem for new job in gb',
                    type=int,
                    default=None)
parser.add_argument('-n', '--nproc',
                    help='processors for new job',
                    type=int,
                    default=None)


args = parser.parse_args()
filename = args.logname
if '.log' not in filename:
    filename += '.log'
outname = args.newname
if outname is None:
    outname = filename[0:filename.index('.')] + '_new'
memreq = args.mem
procreq = args.nproc


def main(filename, outname='test', memreq = None, procreq = None):
    file_lines = []
    # file = open(filename)
    # for line in file:
    #     file_lines.append(line).rstrip('\n')
    with open(filename) as file:
        for line in file:
            file_lines.append(line.rstrip('\n').lstrip(' '))
    geom_out_idcs = []
    geom_out_lines = []
    str_out_idcs = []
    star_idcs = []
    for i, line in enumerate(file_lines):
        if 'Coordinates (Angstroms)' in line:
            geom_out_idcs.append(i)
            geom_out_lines.append(line)
        if 'N-N' in line:
            str_out_idcs.append(i)
        if '************************************' in line:
            star_idcs.append(i)
    last_geom_out_start = max(geom_out_idcs)
    last_geom_out_end = None
    for i, line in enumerate(file_lines):
        if 'Rotational constants' in line:
            last_geom_out_end = i
    assert last_geom_out_end is not None
    xyz_lines = []
    for line in file_lines[last_geom_out_start+3:last_geom_out_end-1]:
        line_split = line.split(' ')
        line_split_clean = []
        for el in line_split:
            if len(el) > 0:
                line_split_clean.append(el)
        xyz_lines.append([line_split_clean[1], line_split_clean[3], line_split_clean[4], line_split_clean[5]])
    last_split = max(str_out_idcs)
    for i, line in enumerate(file_lines[last_split:]):
        if '\\@' in line:
            end_str_dum_idx = i
    str_dump = file_lines[last_split:last_split+end_str_dum_idx+1]
    for i, el in enumerate(str_dump):
        if '#' in el:
            input_line = el
            input_idx = i
    user_input = input_line[input_line.index('#'):(input_line.index('#') + input_line[input_line.index('#'):].index('\\'))]
    slash_idcs = []
    for i, s in enumerate(input_line):
        if s == '\\':
            slash_idcs.append(i)
    title_card = input_line[max(slash_idcs)+1:]
    mem_line = None
    proc_line = None
    for line in file_lines[min(star_idcs):min(star_idcs)+10]:
        if '%mem' in line:
            mem_line = line
        if '%nprocshared' in line.lower() or '%nproc' in line.lower():
            proc_line = line
    save_str = ''
    save_str += '%chk=' + outname + '.chk \n'
    if mem_line is not None:
        save_str += mem_line + ' \n'
    else:
        if memreq is not None:
            save_str += '%mem=' + memreq + 'GB \n'
    if proc_line is not None:
        save_str += proc_line + ' \n'
    else:
        if procreq is not None:
            str_dump += '%NProcShared=' + procreq + ' \n'
    save_str += user_input + ' \n'
    save_str += ' \n'
    save_str += title_card + ' \n'
    for line in str_dump[input_idx+1].split('\\'):
        if len(line.split(',')) == 2:
            ismult = True
            for num in line.split(','):
                if abs(eval(num) - int(eval(num))) == 0:
                    ismult = ismult and True
            if ismult:
                mult_line = line
    save_str += ' \n'
    save_str += mult_line + ' \n'
    for line in xyz_lines:
        save_str += line[0]
        save_str += '        '
        save_str += line[1] + '     '
        save_str += line[2] + '     '
        save_str += line[3] + ' \n'
    save_str += ' \n'
    dump = open(outname + '.gjf', 'w')
    dump.write(save_str)
    dump.close()

if __name__ == '__main__':
    main(filename, outname=outname, memreq=memreq, procreq=procreq)