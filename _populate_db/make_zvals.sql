CREATE TABLE zvals (
    id INTEGER NOT NULL, 
    scan_id INTEGER, 
    val FLOAT, 
    PRIMARY KEY (id), 
    FOREIGN KEY(scan_id) REFERENCES scans (id)
);
