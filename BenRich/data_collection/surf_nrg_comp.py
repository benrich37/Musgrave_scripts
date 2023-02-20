import os
import json
import misc_fns

class surf():
    biases = []
    ecomps = []
    pc = False
    data = {}
    def __init__(self, surf_dir, surf_name, pc=False):
        self.surf_dir = surf_dir
        self.surf_name = surf_name
        self.pc = pc

        self.find_biases()
        self.update_data()
        #self.dump_json()
    def find_biases(self):
        if self.pc:
            #TODO
            return None
        else:
            self.biases = misc_fns.list_dirs(self.surf_dir)
            # biases = os.listdir(self.surf_dir)
            # for b in biases:
            #     if os.path.isdir(os.path.join(self.surf_dir, b)):
            #         self.biases.append(b)
            # print(self.biases)

    def get_ecomp(self, dir):
        out = []
        if self.pc:
            #TODO
            return None
        else:
            file = open(dir + '/Ecomponents')
            for line in file:
                out.append(line.rstrip('\n').replace(' ', '').split('='))
            file.close()
        return out

    def update_data(self):
        for i in range(len(self.biases)):
            self.data[self.biases[i]] = self.get_ecomp(str(self.surf_dir + '/' + self.biases[i]))

    def return_dict(self):
        return self.data

    def dump_json(self):
        with open(os.path.join(self.surf_dir, "data.json"), "w") as outfile:
            json.dump(self.data, outfile)

# os.chdir('/Users/richb/Desktop/Aziz Structures/MNCs/MN4C/pyridinic/backup-2.0/surfs/')
# test = surf('Pt', 'Pt')




