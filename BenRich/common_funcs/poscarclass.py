import numpy as np
import operator
import basis_set_exchange as bse
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
    pc = False
    idk = False
    lats_array = None
    b_per_a = 1.8897259886

    def empty_vars(self):
        self.origin_path = None
        self.origin_directory = None
        self.origin_fname = None
        self.raw_data = []
        self.nameline = None
        self.lat_scale = None
        self.lattice = []
        self.atom_names = None
        self.atom_counts = None
        self.sim_type = None
        self.posns = []
        self.directory = None
        self.sig_figs = None
        self.def_scal_orig = [0.5, 0.5, 0.5]
        self.has_freeze_bools = False
        self.has_atom_ids = True
        self.frac_to_cart_mat = None
        self.cart_to_frac_mat = None
        self.pc = False
        self.idk = False
        self.lats_array = None
        self.selective_dynamics = False
        self.cartesian = False



    def __init__(self, file_path, pc = False):
        self.empty_vars()
        self.pc = pc
        self.origin_path = file_path
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

    def nspace_split(self, list_obj):
        clean = []
        for item in list_obj:
            if len(item) > 0:
                clean.append(item)
        return clean

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
                self.lattice.append(self.nspace_split(line.rstrip().split(' ')))
                continue
            if self.atom_names is None:
                self.atom_names = self.nspace_split(line.rstrip().split(' '))
                continue
            if self.atom_counts is None:
                self.atom_counts = self.nspace_split(line.rstrip().split(' '))
                continue
            if "Selective dynamics" in line:
                self.selective_dynamics = True
                continue
            if "Direct" in line:
                self.cartesian = False
                continue
            elif "Cartesian" in line:
                self.cartesian = True
                continue
            test_line = self.nspace_split(line.rstrip().split(' '))
            if type(test_line) is list and len(test_line) == 1:
                self.idk = test_line[0]
                continue
            self.posns.append(self.nspace_split(line.rstrip().split(' ')))
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
            up_to += int(self.atom_counts[i])
        self.posns = new_posns
        self.has_atom_ids = True

    def organize_origin_strings_pc(self):
        slashes = [0]
        for c in range(self.origin_path.count('\\')):
            slashes.append(self.origin_path[slashes[-1] + 1:].index('\\') + slashes[-1] + 1)
        self.origin_fname = self.origin_path[slashes[-1] + 1:]
        self.origin_directory = self.origin_path[:slashes[-1] + 1]

    def organize_origin_strings_mac(self):
        slashes = [0]
        for c in range(self.origin_path.count('/')):
            slashes.append(self.origin_path[slashes[-1] + 1:].index('/') + slashes[-1] + 1)
        self.origin_fname = self.origin_path[slashes[-1] + 1:]
        self.origin_directory = self.origin_path[:slashes[-1] + 1]

    def organize_origin_strings(self):
        """
        Parses through self.origin_path given in class initialization, finds
        the final present '/', and sets everything before that as the file
        directory and everything after that as the file name
        """
        if self.pc:
            self.organize_origin_strings_pc()
        else:
            self.organize_origin_strings_mac()

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
        if not self.sim_type is None:
            dump_str += self.sim_type + ' \n'
        if self.cartesian:
            dump_str += "Cartesian \n"
        else:
            dump_str += "Direct \n"
        for p in self.posns:
            for x in p:
                dump_str += x + ' '
            dump_str += ' \n'
        return dump_str

    def dump_new_poscar(self, fname):
        new = open(self.origin_directory + fname, 'w')
        new.write(self.get_dump_str())
        new.close()

    def get_xyz_dump_str(self):
        dump_str = ''
        posns_cart = self.get_posns_cart()
        for i in range(len(self.atom_names)):
            for j in range(int(self.atom_counts[i])):
                dump_str += self.atom_names[i] + '   '
                posns = posns_cart[i + j]
                for p in posns[:3]:
                    dump_str += self.float_to_str(p) + ' '
                dump_str += ' \n '
        return dump_str

    def get_xyz_dump_str_alt(self):
        self.append_atom_ids_to_posns()
        dump_str = ''
        posns_cart = self.get_posns_cart()
        for i, p in enumerate(posns_cart):
            dump_str += str(self.posns[i][-1]) + ' '
            for posn in p[:3]:
                dump_str += self.float_to_str(posn) + ' '
            dump_str += '\n'
        # for i in range(len(self.atom_names)):
        #     for j in range(int(self.atom_counts[i])):
        #         dump_str += self.atom_names[i] + '   '
        #         posns = posns_cart[i + j]
        #         for p in posns:
        #             dump_str += self.float_to_str(p) + ' '
        #         dump_str += ' \n '
        return dump_str

    def get_xyz_dump_str_gaussian(self, jobtype='', basis=None, functional='', fname=None, tags=[]):
        assert type(tags) == list
        dump_str2 = ''
        if fname is not None:
            dump_str2 += '%chk=' + fname + '.chk \n'
        dump_str2 += '# ' + jobtype + ' ' + functional + ' ' + 'Gen '
        for t in tags:
            dump_str2 += t + ' '
        dump_str2 += '\n'
        dump_str2 += '\n '
        dump_str2 += 'Title Card Required \n '
        dump_str2 += '\n '
        dump_str2 += '0 1 \n'
        dump_str2 += self.get_xyz_dump_str_alt()
        if basis is not None:
            dump_str2 += '\n'
            dump_str2 += bse.get_basis(basis, elements=self.atom_names, fmt='gaussian94')
        return dump_str2

    def dump_as_xyz(self, fname):
        new = open(self.origin_directory + fname, 'w')
        new.write(self.get_xyz_dump_str())
        new.close()

    def dump_as_gaussian(self, fname, jobtype='', basis=None, functional='', suffix='gjf', tags=[]):
        assert type(tags) == list
        new = open(self.origin_directory + fname + '.' + suffix, 'w')
        new.write(self.get_xyz_dump_str_gaussian(jobtype=jobtype, basis=basis, functional=functional, fname=fname, tags=tags))
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

    # def partial_supercell(self, expand_factors):
    #     """
    #     :param (list[int]) expand_factors: How many images to creates along [a, b, c]
    #     """
    #     self.rescale_unit_cell(expand_factors,
    #                            freeze=True,
    #                            scale_origin=[0., 0., 0.])
    #     x_dup = int(np.ceil(expand_factors[0]))
    #     y_dup = int(np.ceil(expand_factors[1]))
    #     z_dup = int(np.ceil(expand_factors[2]))
    #     new_posns = []
    #     for x in range(x_dup):
    #         for y in range(y_dup):
    #             for z in range(z_dup):
    #                 for p in self.posns:
    #                     new_p = self.translate_posn(x, x_dup,
    #                                                 y, y_dup,
    #                                                 z, z_dup,
    #                                                 p)
    #                     self.app_posn_xtra_data(new_p, p)
    #                     new_posns.append(new_p)
    #     self.posns = new_posns
    #     self.organize_posns_and_headers()
    #     self.cutoff(1.0, 1.0, 1.0)

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

    def get_posns_frac(self):
        assert self.cartesian is False
        new_posns = []
        for p in self.posns:
            tmp_posns = p[0:3]
            flt_posns = []
            for t in tmp_posns:
                flt_posns.append(float(t))
            new_posns.append(flt_posns)
        return new_posns

    def get_lat_vecs_cart(self):
        lat_vecs = [[0,0,0],[0,0,0],[0,0,0]]
        for i in range(3):
            for j in range(3):
                lat_vecs[j][i] = float(self.lattice[i][j])*float(self.lat_scale)
        return lat_vecs

    def get_posns_cart(self):
        if self.cartesian:
            return self.posns
        else:
            posns = self.get_posns_frac()
            lats = self.get_lat_vecs_cart()
            posns_cart = []
            for p in posns:
                sum_hold = np.zeros(3)
                for i in range(3):
                    sum_hold += p[i]*np.array(lats[i])
                posns_cart.append(sum_hold)
            return posns_cart

    def get_approx_atom_center_idx(self, grid_dims, atom_idx, atom_type = None):
        posn = self.posns[atom_idx]
        if not atom_type is None:
            assert posn[-1] == atom_type
        center_idx = []
        for i in range(3):
            center_idx.append(int(float(posn[i])*float(grid_dims[i])))
        return center_idx

    def get_dists_in_idcs(self, grid_dims, atom_idx, atom_type = None):
        center_idx = self.get_approx_atom_center_idx(grid_dims, atom_idx, atom_type=atom_type)
        center_idcs = []
        for i in range(len(self.posns)):
            center_idcs.append(self.get_approx_atom_center_idx(grid_dims, i))
        dists = []
        for idx in center_idcs:
            dists.append(np.linalg.norm(np.array(center_idx) - np.array(idx)))
        return dists

    def get_conservative_atom_radius(self, grid_dims, atom_idx, atom_type = None):
        dists = self.get_dists_in_idcs(grid_dims, atom_idx, atom_type=atom_type)
        dists.sort()
        return int(dists[1])


    def _jdftx_ionpos_dumpstr(self, posns, cart):
        if cart:
            dump_str = "coords-type Cartesian \n"
        else:
            dump_str = "coords-type Lattice \n"
        for i in range(len(posns)):
            dump_str += "ion " + self.posns[i][-1] + " "
            for j in range(3):
                dump_str += self.float_to_str(posns[i][j]*self.b_per_a) + " \t"
            dump_str += " 1 \n"
        return dump_str

    def dump_jdftx_ionpos(self, fname = None, cart = True):
        if fname is None:
            fname = "params.ionpos"
        if cart:
            posns = self.get_posns_cart()
        else:
            posns = self.get_posns_frac()
        dump_str = self._jdftx_ionpos_dumpstr(posns, cart)
        with open(self.origin_directory + fname, 'w') as f:
            f.write(dump_str)

    def _jdftx_lattice_dumpstr(self, lat):
        dump_str = "lattice \\ \n"
        for i in range(2):
            for j in range(3):
                dump_str += "\t" + self.float_to_str(lat[i][j]*self.b_per_a)
            dump_str += " \\ \n"
        for j in range(3):
            dump_str +="\t" + self.float_to_str(lat[2][j]*self.b_per_a)
        dump_str += " \n"
        return dump_str

    def dump_jdftx_lattice(self, fname = None):
        if fname is None:
            fname = "params.lattice"
        lat = self.get_lat_vecs_cart()
        dump_str = self._jdftx_lattice_dumpstr(lat)
        with open(self.origin_directory + fname, 'w') as f:
            f.write(dump_str)



    # def get_grid_centers_cart(self, grid_dims):
    #     lats = self.get_lat_vecs_cart()
    #     lats_array = [np.array(lats[0]), np.array(lats[1]), np.array(lats[2])]
    #     centers = []
    #     for i in range(grid_dims[0]):
    #         centers.append([])
    #         for j in range(grid_dims[1]):
    #             centers[-1].append([])
    #             for k in range(grid_dims[2]):
    #                 center = ((i + 0.5)/grid_dims[0])*lats_array[0] + ((j + 0.5)/grid_dims[1])*lats_array[1] + ((k + 0.5)/grid_dims[2])*lats_array[2]
    #                 centers[-1][-1].append(center)
    #     return centers
    #
    # def get_nearest_atom(self, posn, cart_posns):
    #     posn = np.array(posn)
    #     cur_best = 100
    #     cur_idx = 0
    #     for i in range(len(cart_posns)):
    #         dist = np.linalg.norm(posn - cart_posns[i])
    #         if dist < cur_best:
    #             cur_best = dist
    #             cur_idx = i
    #     return cur_idx, cur_best
    #
    # def get_grid_nearest_atoms(self, grid_dims, cutoff = 1.0):
    #     """ For grid dims (i, j, k), returns an i by j by k list of list of lists, where element i,j,k contains
    #     None or the atom index of its nearest neighbor
    #     :param grid_dims:
    #     :param cutoff:
    #     :return:
    #     """
    #     grid_centers = self.get_grid_centers_cart(grid_dims)
    #     cart_posns = self.get_posns_cart()
    #     claimed_dict = {}
    #     grid_idcs = []
    #     for i in range(grid_dims[0]):
    #         grid_idcs.append([])
    #         for j in range(grid_dims[1]):
    #             grid_idcs[-1].append([])
    #             for k in range(grid_dims[2]):
    #                 idx, dist = self.get_nearest_atom(
    #                     grid_centers[i][j][k],
    #                     cart_posns
    #                 )
    #                 if dist < cutoff:
    #                     append_value = idx
    #                 else:
    #                     append_value = None
    #                 grid_idcs[-1][-1].append(append_value)
    #     return grid_idcs
    #
    # def get_atoms_encompassed_grid_pts(self, grid_dims, cutoff=1.0):
    #     """ Returns a dictionary where key values are the atom index, and dict holds the i/j/k indices of grid points
    #     encompassed by atom
    #     :param grid_dims:
    #     :param cutoff:
    #     :return:
    #     """
    #     encompass_dict = {}
    #     for i in range(len(self.posns)):
    #         encompass_dict[i] = []
    #     grid_idcs = self.get_grid_nearest_atoms(grid_dims, cutoff=cutoff)
    #     for i in range(len(grid_idcs[0])):
    #         for j in range(len(grid_idcs[1])):
    #             for k in range(len(grid_idcs[2])):
    #                 if grid_idcs[i][j][k] is not None:
    #                     encompass_dict[grid_idcs[i][j][k]].append(tuple([i,j,k]))
    #     return encompass_dict

