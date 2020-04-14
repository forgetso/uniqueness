import numpy as np
from backend.settings import SEMANTIC_SPACE_FOLDER
from nilearn import surface
from nilearn import datasets


# TODO import this data to the database ??

class BrainMap():

    def __init__(self):
        semantic_space_data = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
        self.english1000_vocab = semantic_space_data['english1000_vocab']
        self.english1000_embedding = semantic_space_data['english1000_embedding']
        self.pragmatic_info = np.load('{}/pragmatic-info.npz'.format(SEMANTIC_SPACE_FOLDER))
        self.lsemwts = self.pragmatic_info['lsemwts']
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

    def get_semantic_area(self, word):
        result = ''
        found = False
        byte_word = bytes(word, encoding="ascii")
        if np.where(self.english1000_vocab == byte_word)[0].size != 0:
            w = np.vstack([self.english1000_embedding[:, np.where(self.english1000_vocab == byte_word)[0][0]]]).mean(0)
            # the predicted response of each of the 985 features to the list of words
            response = np.dot(w, self.lsemwts)
            # The index area in which the word had the highest response
            index = np.where(response == response.max())
            # label assigned to area
            label = self.pragmatic_info['lnames'][index]
            # is this a significant area? True or False
            if self.pragmatic_info['lsignif'][index][0]:
                result = label[0].decode('utf-8')
                found = True
        return found, result

    def get_semantic_coordinates(self, label):
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
