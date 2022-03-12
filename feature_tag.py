from collections import OrderedDict, namedtuple, defaultdict

DEFAULT_GROUP_NAME = "default_group"

class SparseFeat(namedtuple(typename='SparseFeat',
                            field_names=['name', 'vocabulary_size', 'embedding_dim',
                                        'use_hash', 'dtype', 'embedding_name', 'group_name'])):
    # feature tag with fields recording the properties of this feature
    __slots__ = ()

    def __new__(cls, name,
                vocabulary_size,
                embedding_dim=4, use_hash=False,
                dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):

        if embedding_name is None:
            embedding_name = name
        # if embedding_dim == "auto":
        #     embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        return super(SparseFeat, cls).__new__(cls,
                                              name,
                                              vocabulary_size,
                                              embedding_dim,
                                              use_hash,
                                              dtype,
                                              embedding_name,
                                              group_name)

    def __hash__(self):
        # The only required property is that objects which compare equal have the same hash value
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple(typename='VarLenSparseFeat',
                                  field_names=['sparsefeat', 'maxlen', 'combiner',
                                              'length_name', 'weight_name', 'weight_norm'])):
    # used in run_din.py
    # does not inherit SparseFeat
    __slots__ = ()

    def __new__(cls, sparsefeat,
                maxlen, combiner="mean",
                length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls,
                                                    sparsefeat,
                                                    maxlen,
                                                    combiner,
                                                    length_name,
                                                    weight_name,
                                                    weight_norm)

    # all of these properties are fields of the field sparsefeat
    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple(typename='DenseFeat',
                           field_names=['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()