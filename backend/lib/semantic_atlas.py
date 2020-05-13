import numpy as np
from backend.settings import SEMANTIC_SPACE_FOLDER
from nilearn import surface
from nilearn import datasets


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
        for hemi in self.hemi_data:
            labels = self.pragmatic_info['{}names'.format(hemi[0])]
            vert = self.hemi_data[hemi]['semantic']['parc']
            rr, _ = surface.load_surf_mesh(fsaverage['pial_{}'.format(hemi)])
            for k, label in enumerate(labels):
                if label != b'':  # Omit the Unknown label.
                    # Compute mean location of vertices in label of index k
                    coordinates[hemi][label.decode('utf-8')] = np.mean(rr[vert == k], axis=0)
        # coordinates = np.array(coordinates)  # 3D coordinates of parcels
        return coordinates

    def get_semantic_area(self, words):
        # get the words in the semantic model that have been given as WA responses (10470,) True or False
        words_found = np.isin(self.english1000_vocab, words)
        # filter the 985 word embedding vectors to those words only (985 x n words)
        words_regions = self.english1000_embedding[:, words_found]
        # - average semantic VM weights within each area and across subjects
        # `lsemwts` (985 features x 192 areas), `rsemwts` (985 features x 128 areas)
        # If `v` is a shape `(985,)` vector of weights and `english1000_embedding` is the shape `(985, 10470)` embedding matrix,
        # then `np.dot(v, english1000_embedding)` yields a vector of shape `(10470,)` containing the predicted response to each of the 10470 words in the vocabulary.
        w = np.vstack([self.english1000_embedding[:, words_found]])
        # the predicted response of each of the 985 features to the list of words
        # the transpose of w. Giving shapes (n,985) and (985,192). This will need to be put in np vectorize or similar.
        response_left = np.dot(w.T, self.lsemwts)
        response_right = np.dot(w.T, self.rsemwts)
        # get the indices of the labels that the regions correspond to
        r_indices = np.vstack(np.where(response_right == response_right.max()))
        l_indices = np.vstack(np.where(response_left == response_left.max()))
        # l_significant = pragmatic_info['lsignif'][l_indices]
        l_labels = self.pragmatic_info['lnames'][l_indices]
        # r_significant = pragmatic_info['rsignif'][r_indices]
        r_labels = self.pragmatic_info['rnames'][r_indices]
        # if the regions are significant then the labels are present, otherwise they are an empty byte string
        return l_labels, r_labels

    def get_semantic_coordinates(self, label):
        # TODO right AND left hemisphere
        coordinates = None
        for hemi in self.hemi_data:
            if label in self.coordinates[hemi]:
                coordinates = self.coordinates[hemi][label]
        return coordinates

    def get_distances_between_semantic_areas(self, area_x, area_y):
        cue_coords = self.get_semantic_coordinates(area_x)
        response_coords = self.get_semantic_coordinates(area_y)
        return self.get_distance(cue_coords, response_coords)

    def get_distance(self, coord_x, coord_y):
        return np.sqrt(np.sum((coord_x - coord_y) ** 2))
