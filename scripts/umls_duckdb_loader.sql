
CREATE TABLE dim_VOCABULARY AS
SELECT * FROM read_csv('./data/VOCABULARY.csv', delim = '\t');
