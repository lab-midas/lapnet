from pyexcel_ods import get_data


def read_fnames(fnames_path):
    infile = open(fnames_path, 'r')
    contents = infile.read().strip().split()
    data_paths = [f for f in contents]
    infile.close()
    return data_paths


def get_slice_info_from_ods_file(info_file):
    ods = get_data(info_file)
    slice_info = {value[0]: list(range(*[int(j) for j in value[1].split(',')])) for value in ods["Sheet1"] if
                  len(value) != 0}
    return slice_info


def get_maxmin_info_from_ods_file(info_file):
    ods = get_data(info_file)
    slice_info = {value[0]: list(value[1].split(',')) for value in ods["Sheet1"] if len(value) != 0}
    return slice_info


class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]