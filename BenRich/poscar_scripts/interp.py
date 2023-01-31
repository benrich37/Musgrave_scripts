import numpy as np
import operator
ops = {
    '>': operator.gt,
    '<': operator.lt
}

class Poscar():
    origin_path = None
    origin_directory = None
    origin_fname = None
    raw_data = []
    nameline = None
    lat_scale = None
    lattice = []
    atom_names = None
    atom_counts = None
    sim_type = None
    posns = []
    directory = None
    sig_figs = None
    def_scal_orig = [0.5, 0.5, 0.5]
    has_freeze_bools = False
    has_atom_ids = True
    frac_to_cart_mat = None
    cart_to_frac_mat = None


    def __init__(self, file_path):
        self.origin_path = file_path
        self.raw_data = []
        for line in open(file_path):
            self.raw_data.append(line.removesuffix('\n'))
        self.organize_data()
        self.organize_origin_strings()
        self.get_sig_figs()
        if not self.has_atom_ids:
            self.append_atom_ids_to_posns()
        if not self.has_freeze_bools:
            self.freeze_all()
        self.update_projection_mats()

    def is_whitespace_line(self, line):
        return type(line == list) and len(line) == 0

    def pop_whitespace_lines(self):
        """
        Edits the raw data to remove any empty lines
        """
        bad_idcs = []
        for i in range(len(self.raw_data)):
            if self.is_whitespace_line(self.raw_data[i]):
                bad_idcs.append(i)
        new_raw_data = []
        for i in range(len(self.raw_data)):
            if i not in bad_idcs:
                new_raw_data.append(self.raw_data[i])
        self.raw_data = new_raw_data

    def organize_data(self):
        """
        Parses through the raw_data (a list of strings, where each string is
        a line in the given file)
        Expects the following order of information:
            (1) nameline
            (2) lat_scale
            (3, 4, 5) lattice
            (6) atom_names (following order of appearance)
            (7) atom_counts (following order of atom names)
            (8) sim_type (idk)
            (9 - end) posns
                x y z T/F T/F T/F XX (has_freeze_bools and has_atom_ids)
                x y z XX (has_atom_ids)
                x y z T/F T/F T/F (has_freeze_bools)
                x y z ()
        """
        self.pop_whitespace_lines()
        for line in self.raw_data:
            if self.nameline is None:
                self.nameline = line
                continue
            if self.lat_scale is None:
                self.lat_scale = line
                continue
            if len(self.lattice) < 3:
                self.lattice.append(line.rstrip().split(' '))
                continue
            if self.atom_names is None:
                self.atom_names = line.rstrip().split(' ')
                continue
            if self.atom_counts is None:
                self.atom_counts = line.rstrip().split(' ')
                continue
            if self.sim_type is None:
                self.sim_type = line
                continue
            self.posns.append(line.rstrip().split(' '))
        if len(self.posns[0]) == 3 or len(self.posns[0]) == 6:
            self.has_atom_ids = False
        if len(self.posns[0]) > 4:
            self.has_freeze_bools = True

    def append_atom_ids_to_posns(self):
        """
        Goes through the atom_counts line ([x0, x1, x2 ...]), assumes
        self.posns[0:x0] are atom type self.atom_names[0],
        self.posns[x0:x0+x1] are of atom type self.atom_names[1],
        and so on
        (self.posns[(...)+(xn-1):(...)+(xn-1)+(xn)] are of atom type self.atom_atoms[n])
        """
        new_posns = []
        up_to = 0
        for i in range(len(self.atom_counts)):
            for j in range(int(self.atom_counts[i])):
                new_p = []
                hold_p = self.posns[up_to + j]
                for x in hold_p[:3]:
                    new_p.append(x)
                if self.has_freeze_bools:
                    for x in hold_p[3:6]:
                        new_p.append(x)
                new_p.append(self.atom_names[i])
                new_posns.append(new_p)
            up_to += self.atom_counts[i]
        self.posns = new_posns
        self.has_atom_ids = True

    def organize_origin_strings(self):
        """
        Parses through self.origin_path given in class initialization, finds
        the final present '/', and sets everything before that as the file
        directory and everything after that as the file name
        """
        slashes = [0]
        for c in range(self.origin_path.count('/')):
            slashes.append(self.origin_path[slashes[-1] + 1:].index('/') + slashes[-1] + 1)
        self.origin_fname = self.origin_path[slashes[-1] + 1:]
        self.origin_directory = self.origin_path[:slashes[-1] + 1]

    def get_sig_figs(self):
        """
        Gets the sig figs expected in the POSCAR file
        """
        sample = self.lattice[0][0]
        self.sig_figs = len(sample) - sample.count('.')

    def get_dump_str(self):
        dump_str = ''
        dump_str += self.nameline + ' \n'
        dump_str += self.lat_scale + ' \n'
        for line in self.lattice:
            for num in line:
                dump_str += num + ' '
            dump_str += ' \n'
        for a in self.atom_names:
            dump_str += a + ' '
        dump_str += ' \n'
        for n in self.atom_counts:
            dump_str += n + ' '
        dump_str += ' \n'
        dump_str += self.sim_type + ' \n'
        for p in self.posns:
            for x in p:
                dump_str += x + ' '
            dump_str += ' \n'
        return dump_str

    def dump_new_poscar(self, fname):
        new = open(self.origin_directory + fname, 'w')
        new.write(self.get_dump_str())
        new.close()

    def float_to_str(self, num):
        sig1 = len(str(int(num)))
        sig2 = self.sig_figs - sig1
        arg = '%.' + str(sig2) + 'f'
        return arg % num

    def get_new_lat_posn(self, posn, scale_origin, scale_factors):
        ds = []
        for i in range(len(scale_origin)):
            d_in_new_scale = (float(posn[i]) - scale_origin[i])/(scale_factors[i])
            ds.append(self.float_to_str(d_in_new_scale + scale_origin[i]))
        return ds

    def translate_posn(self, x, xmax, y, ymax, z, zmax, posn):
        x0 = float(posn[0])
        y0 = float(posn[1])
        z0 = float(posn[2])
        new_posn = [x0 + (x/xmax), y0 + (y/ymax), z0 + (z/zmax)]
        for i in range(len(new_posn)):
            new_posn[i] = self.float_to_str(new_posn[i])
        return new_posn

    def rescale_unit_cell(self, scale_factors, freeze = True, scale_origin = None):
        """
        :param (list[float]) scale_factors: List of 3 floats to rescale unit cell
        :param (bool) freeze: Freezes atoms to keep distances wrt scale_origin
        instead of rescaling with unit cell
        :param scale_origin: Optional provided scale_origin
        (default is self.def_scal_orig)
        """
        cx, cy, cz = scale_factors[0], scale_factors[1], scale_factors[2]
        lat_mat = np.array(self.lattice, dtype=float)
        scale_mat = np.array([[cx,0,0], [0,cy,0], [0,0,cz]])
        prod_mat = np.dot(lat_mat, scale_mat)
        save_mat = []
        for x in prod_mat:
            app = []
            for y in x:
                app.append(self.float_to_str(y))
            save_mat.append(app)
        if freeze:
            if scale_origin is None:
                scale_origin = self.def_scal_orig
            new_posns = []
            for p in self.posns:
                new_p = self.get_new_lat_posn(p, scale_origin, scale_factors)
                self.app_posn_xtra_data(new_p, p)
                new_posns.append(new_p)
            self.posns = new_posns
        self.lattice = save_mat
        self.update_projection_mats()

    def update_projection_mats(self):
        """
        Creates matrices which transforms a positional vector to/fro
        fractional/cartesian coordinates upon taking the dot product
        """
        hold_mat1 = np.zeros(tuple([3,3]))
        hold_mat2 = np.zeros(tuple([3,3]))
        for v in self.lattice:
            v_use = np.array(v, dtype=float)
            out_v = np.outer(v_use,v_use)
            hold_mat1 += out_v/np.linalg.norm(v_use)
            hold_mat2 += out_v/(np.linalg.norm(v_use)**3)
        self.frac_to_cart_mat = hold_mat1
        self.cart_to_frac_mat = hold_mat2


    def organize_posns_and_headers(self):
        new_atom_types = []
        new_atom_counts = []
        org_posns = []
        for p in self.posns:
            atom = p[-1]
            if atom not in new_atom_types:
                new_atom_types.append(atom)
                new_atom_counts.append(0)
                org_posns.append([])
            idx = new_atom_types.index(atom)
            new_atom_counts[idx] = int(new_atom_counts[idx] + 1)
            org_posns[idx].append(p)
        for i in range(len(new_atom_counts)):
            new_atom_counts[i] = str(new_atom_counts[i])
        self.atom_names = new_atom_types
        self.atom_counts = new_atom_counts
        self.posns = []
        for i in org_posns:
            for j in i:
                self.posns.append(j)

    def supercell(self, expand_factors):
        """
        :param (list[int]) expand_factors: How many images to creates along [a, b, c]
        """
        self.rescale_unit_cell(expand_factors,
                               freeze=True,
                               scale_origin=[0., 0., 0.])
        x_dup = int(expand_factors[0])
        y_dup = int(expand_factors[1])
        z_dup = int(expand_factors[2])
        new_posns = []
        for x in range(x_dup):
            for y in range(y_dup):
                for z in range(z_dup):
                    for p in self.posns:
                        new_p = self.translate_posn(x, x_dup,
                                                    y, y_dup,
                                                    z, z_dup,
                                                    p)
                        self.app_posn_xtra_data(new_p, p)
                        new_posns.append(new_p)
        self.posns = new_posns
        self.organize_posns_and_headers()

    def cutoff_check(self, posn, x, y, z, op_x, op_y, op_z):
        bad_x = op_x(float(posn[0]), x)
        bad_y = op_y(float(posn[1]), y)
        bad_z = op_z(float(posn[2]), z)
        bad_posn = (bad_x or bad_y) or bad_z
        return bad_posn

    def cutoff(self, x, y, z, op_key_x = '>', op_key_y = '>', op_key_z = '>'):
        op_x = ops[op_key_x]
        op_y = ops[op_key_y]
        op_z = ops[op_key_z]
        new_posns = []
        for p in self.posns:
            if not self.cutoff_check(p, x, y, z, op_x, op_y, op_z):
                new_posns.append(p)
        self.posns = new_posns
        self.organize_posns_and_headers()

    def mean_posn(self):
        sum = np.zeros(3)
        for p in self.posns:
            sum += np.array(p[:3], dtype=float)
        return sum / len(self.posns)

    def app_posn_xtra_data(self, con_posn, posn):
        if self.has_freeze_bools:
            for x in posn[-4:]:
                con_posn.append(x)
        else:
            con_posn.append(posn[-1])

    def center_posns(self):
        mean = self.mean_posn()
        new_posns = []
        for p in self.posns:
            new_p = list(np.array(p[:3], dtype=float) - mean + np.array([0.5, 0.5, 0.5]))
            for i in range(len(new_p)):
                new_p[i] = self.float_to_str(new_p[i])
            self.app_posn_xtra_data(new_p, p)
            new_posns.append(new_p)
        self.posns = new_posns

    def extrema_atoms(self, a = True, b = True, c = True):
        # all_dists = []
        all_sums = []
        ref_pt = self.def_scal_orig
        ref_bools = [a, b, c]
        for p in self.posns:
            # dists_p = []
            sum_hold = 0
            for i in range(len(ref_bools)):
                if ref_bools[i]:
                    num = abs(float(p[i]) - ref_pt[i])
                    # dists_p.append(num)
                    sum_hold += num
            # all_dists.append(dists_p)
            all_sums.append(sum_hold**2)
        return all_sums

    def freeze_all(self):
        if self.has_freeze_bools:
            for i in range(len(self.posns)):
                for j in range(3):
                    self.posns[i][3 + j] = 'F'
        else:
            if self.has_atom_ids:
                for i in range(len(self.posns)):
                    new_p = []
                    for x in self.posns[i][:3]:
                        new_p.append(x)
                    for j in range(3):
                        new_p.append('F')
                    new_p.append(self.posns[i][-1])
                    self.posns[i] = new_p
            else:
                for i in range(len(self.posns)):
                    new_p = []
                    for x in self.posns[:3]:
                        new_p.append(x)
                    for j in range(3):
                        new_p.append('F')
                    self.posns[i] = new_p
            self.has_freeze_bools = True

    def outside_sphere(self, a_b_c, radius, origin):
        check = 0
        for i in range(3):
            check += (a_b_c[i] - origin[i])**2
        return np.sqrt(check) > radius

    def outside_cylinder(self, a_b_c, radius, origin, dir_vec, eval_in_cart = False):
        origin = np.array(origin)
        dir_vec = np.array(dir_vec)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        vecs = np.linalg.eig(np.eye(3) - np.outer(dir_vec, dir_vec))[1]
        orth_vecs = []
        for v in vecs.T:
            if np.dot(v, dir_vec) < 0.1:
                orth_vecs.append(v/np.linalg.norm(v))
        a_b_c_wrt_o = np.array(a_b_c, dtype=float) - origin
        if eval_in_cart:
            a_b_c_wrt_o = np.dot(self.frac_to_cart_mat, a_b_c_wrt_o)
        check = 0
        for i in range(2):
            check += np.dot(a_b_c_wrt_o, orth_vecs[i])**2
        return np.sqrt(check) > radius

    def count_by_lambda_booler(self, f_a_b_c):
        got_true = []
        for i in range(len(self.posns)):
            if f_a_b_c(self.posns[i][:3]):
                got_true.append(i)
        return got_true

    def unfreeze_by_lambda_booler(self, f_a_b_c, a = True, b = True, c = True,
                                  init_frozen = False):
        """
        :param f_a_b_c: Function which acts on a list of three floats
        :param a:
        :param b:
        :param c:
        :return:
        """
        ref_bools = [a, b, c]
        if not self.has_freeze_bools or init_frozen:
            self.freeze_all()
        for i in range(len(self.posns)):
            if f_a_b_c(self.posns[i][:3]):
                for j in range(3):
                    if ref_bools[j]:
                        self.posns[i][3 + j] = 'T'