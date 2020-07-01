# build a table of all words and their measures

query = """SELECT
                    
                    LOWER(a.lemma_highest_pos) as word_lemma,
                    a.aoa_kup_lemma as age,
                    a.dom_pos_subtlex as part_of_speech,
                    LOWER(c.word) AS c_word,
                    LOWER(a.word) AS a_word,
                    LOWER(o.word) AS o_word,
                    LOWER(e.word) AS e_word,
                    LOWER(phon.word) as phon_word,
                    e.arousal AS arousal,
                    e.dominance AS dominance,
                    e.valence AS valence,
                    o.bigram_count AS bigram_count,
                    o.bigram_freq AS bigram_freq,
                    o.frequency AS frequency,
                    o.ortho_frequency AS ortho_frequency,
                    o.ortho_neighbours AS ortho_neighbours,
                    o.trigram_count AS trigram_count,
                    o.trigram_freq AS trigram_freq,
                    o.unconstrained_bigram_count AS unconstrained_bigram_count,
                    o.unconstrained_bigram_freq AS unconstrained_bigram_freq,
                    o.unconstrained_trigram_count AS unconstrained_trigram_count,
                    o.unconstrained_trigram_freq AS unconstrained_trigram_freq,
                    o.unconstrained_unigram_count AS unconstrained_unigram_count,
                    o.unconstrained_unigram_freq AS unconstrained_unigram_freq,
                    o.unigram_count AS unigram_count,
                    o.unigram_freq AS unigram_freq,
                    p.pknown AS pknown,
                    p.prevalence AS prevalence,
                    phon.nphon AS nphon,
                    phon.scdcnt AS scdcnt,
                    phon.sfreq AS sfreq,
                    phon.strbpav AS strbpav,
                    phon.strdens AS strdens,
                    phon.strlcpospav AS strlcpospav,
                    phon.strpospav AS strpospav,
                    phon.strtpav AS strtpav,
                    phon.unsbpav AS unsbpav,
                    phon.unsdens AS unsdens,
                    phon.unslcpospav AS unslcpospav,
                    phon.unspospav AS unspospav,
                    phon.unstpav AS unstpav,
                    c.concreteness_measure AS concreteness_measure
                FROM orthographic o
                FULL OUTER JOIN concreteness c ON o.word=c.word
                FULL OUTER JOIN prevalences p ON o.word=p.word
                FULL OUTER JOIN emotions e ON o.word=e.word
                FULL OUTER JOIN acquisition_age a ON o.word=a.word
                FULL OUTER JOIN phonographic phon ON o.word=phon.word
                ORDER BY c_word, a_word, o_word, e_word, phon_word  ASC;"""
