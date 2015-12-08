import logging
from codify.config.settings import *
import numpy as np

class WordVectors:
    def __init__(self):
        self.logger = logging.getLogger(LOGGER)
        self.embeddings_path = GLOVE_DIR + "/" + EMBEDDIGS_FILE
        self.embeddings = {}

    def set_embeddings(self):
        self.logger.info('getting embeddings ...')

        f = file(self.embeddings_path)

        line_count = 1
        for line in f:
            if line_count % 10000 == 0: self.logger.info('processed %d words' % line_count)
            word = line.split()[0]
            embedding = np.array(map(float, line.split()[1:]))
            self.embeddings[word] = embedding

            line_count += 1

        f.close()