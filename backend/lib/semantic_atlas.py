import numpy as np
from backend.settings import SEMANTIC_SPACE_FOLDER
from nilearn import surface
from nilearn import datasets
import pandas as pd

UNKNOWN_AREA_LABEL = b'U'


# TODO import this data to the database ??

# TODO Even when restricting the algorithm to words used in the Gallant vocab, roughly 12% of words do not return a semantic area.
#   Is this because they are not statistically significant in a voxel?


class BrainMap():

    def __init__(self):
        semantic_space_data = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
        self.english1000_vocab = semantic_space_data['english1000_vocab']
        self.english1000_embedding = semantic_space_data['english1000_embedding']
        self.pragmatic_info = np.load('{}/pragmatic-info.npz'.format(SEMANTIC_SPACE_FOLDER))
        self.lsemwts = self.pragmatic_info['lsemwts']
        self.rsemwts = self.pragmatic_info['rsemwts']
        pragmatic_semantic_data_left = np.load('{}/lh.pragmatic-semantic.npz'.format(SEMANTIC_SPACE_FOLDER))
        pragmatic_semantic_data_right = np.load('{}/rh.pragmatic-semantic.npz'.format(SEMANTIC_SPACE_FOLDER))
        self.hemi_data = {'left': {'semantic': pragmatic_semantic_data_left, 'rr': None},
                          'right': {'semantic': pragmatic_semantic_data_right, 'rr': None}}
        self.coordinates = self.get_label_coordinates()

    def get_label_coordinates(self):
        # Retrieve fsaverage5 surface dataset for the plotting background. It contains
        # the surface template as pial and inflated version and a sulcal depth maps
        # which is used for shading
        # high resolution fsaverage with 163842 nodes (https://nilearn.github.io/modules/generated/nilearn.datasets.fetch_surf_fsaverage.html)
        fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
        coordinates = {'left': {}, 'right': {}}

        for hemi in coordinates.keys():
            labels = self.pragmatic_info['{}names'.format(hemi[0])]
            vert = self.hemi_data[hemi]['semantic']['parc']
            rr, _ = surface.load_surf_mesh(fsaverage['pial_{}'.format(hemi)])
            # (192,) labels
            # (163842,) vert (all of the hemisphere vertices labelled 0-191 (or 0-127)
            # (163842, 3) rr coordinates of all of the vertices in high resolution fsaverage
            # append the vertices labels to the vertices coordinates
            coords = pd.DataFrame(np.c_[rr, vert], columns=['coord_x', 'coord_y', 'coord_z', 'label'])
            # take the mean of the coordinates by the labels to get a naive centre point
            coordinates[hemi] = coords.groupby('label').mean()
            coordinates[hemi].index = labels
            coordinates[hemi].index = coordinates[hemi].index.rename('label')
        return coordinates

    def get_semantic_area(self, words):
        # get the words in the semantic model that have been given as WA responses (10470,) True or False
        words_found = np.isin(self.english1000_vocab, words)

        # words_regions = self.english1000_embedding[:, words_found]
        # - average semantic VM weights within each area and across subjects
        # `lsemwts` (985 features x 192 areas), `rsemwts` (985 features x 128 areas)
        # If `v` is a shape `(985,)` vector of weights and `english1000_embedding` is the shape `(985, 10470)` embedding matrix,
        # then `np.dot(v, english1000_embedding)` yields a vector of shape `(10470,)` containing the predicted response to each of the 10470 words in the vocabulary.
        # filter the 985 word embedding vectors to those words only (985 x n words)
        w = np.vstack([self.english1000_embedding[:, words_found]])
        # the predicted response of each of the 985 features to the list of words
        # the transpose of w. Giving shapes (n,985) and (985,192). This will need to be put in np vectorize or similar.
        response_left = np.dot(w.T, self.lsemwts)
        response_right = np.dot(w.T, self.rsemwts)
        # get the indices of the labels that the regions correspond to
        r_indices = np.vstack(np.argmax(response_right, axis=1))
        l_indices = np.vstack(np.argmax(response_left, axis=1))
        # l_significant = pragmatic_info['lsignif'][l_indices]
        l_labels = self.pragmatic_info['lnames'][l_indices]
        # r_significant = pragmatic_info['rsignif'][r_indices]
        r_labels = self.pragmatic_info['rnames'][r_indices]
        # if the regions are significant then the labels are present, otherwise they are an empty byte string
        words_subset = self.english1000_vocab[words_found]
        labels = pd.DataFrame({'word': words_subset,
                               'left_hemi_label': l_labels.flatten(),
                               'right_hemi_label': r_labels.flatten()})
        return labels

    def get_semantic_coordinates(self, hemi, labels):
        if hemi not in self.hemi_data:
            raise KeyError('hemi must be one of {}'.format(self.hemi_data.keys()))
        coordinates_intersect = np.intersect1d(labels, self.coordinates[hemi].index)
        coordinates = np.c_[
            coordinates_intersect,
            self.coordinates[hemi][self.coordinates[hemi].index.isin(coordinates_intersect)]
        ]
        coordinates = pd.DataFrame(coordinates, columns=['label', 'coord_x', 'coord_y', 'coord_z']).set_index('label')
        return coordinates

    def get_distances_between_semantic_areas(self, hemi, area_x, area_y):
        x_coords = self.get_semantic_coordinates(hemi, area_x.values)
        y_coords = self.get_semantic_coordinates(hemi, area_y.values)
        area_x_df = pd.DataFrame(area_x).merge(x_coords, left_on=area_x.name, right_index=True)
        area_y_df = pd.DataFrame(area_y).merge(y_coords, left_on=area_y.name, right_index=True)
        df = area_x_df.join(area_y_df, lsuffix='_left', rsuffix='_right').dropna()
        coord_x = [df['coord_x_left'], df['coord_y_left'], df['coord_z_left']]
        coord_y = [df['coord_x_right'], df['coord_y_right'], df['coord_z_right']]
        df['{}_hemi_distance'.format(hemi)] = self.get_distance(coord_x, coord_y)
        return df

    def get_distance(self, coord_a, coord_b):
        return np.sqrt(np.sum((np.subtract(coord_a, coord_b)) ** 2, axis=0).astype(float))
