
DB_SCHEME = ["""
CREATE TABLE Compartment (
  code VARCHAR(64),
  name VARCHAR(128),
  color VARCHAR(8),
  PRIMARY KEY (code)     
);
""",
"""
CREATE TABLE ExpList (
  code VARCHAR(64),
  name VARCHAR(128),  
  PRIMARY KEY (code)     
);
""",
"""
CREATE TABLE DataProjection (
  code VARCHAR(64),
  name VARCHAR(128),  
  PRIMARY KEY (code)     
);
""",
"""
CREATE TABLE Protein (
  pid VARCHAR(32),
  alt_ids TEXT,
  description TEXT,
  gene_name VARCHAR(32) NOT NULL,
  train_organelle VARCHAR(64),
  suborganelle VARCHAR(64),
  singleness FLOAT,
  likely_single INT,
  pred_class1 VARCHAR(64),
  pred_class2 VARCHAR(64),
  pred_class3 VARCHAR(64),
  pred_text VARCHAR(128),
  p_val FLOAT,
  p_val_single FLOAT,
  p_val_dual FLOAT,
  completeness FLOAT,
  novelty FLOAT,
  FOREIGN KEY (train_organelle) REFERENCES Compartment(code),
  FOREIGN KEY (suborganelle) REFERENCES Compartment(code),
  FOREIGN KEY (pred_class1) REFERENCES Compartment(code),
  FOREIGN KEY (pred_class2) REFERENCES Compartment(code),
  FOREIGN KEY (pred_class2) REFERENCES Compartment(code),   
  PRIMARY KEY (pid)      
);
""",
"""
CREATE TABLE CompartmentScore (
  compartment VARCHAR(64),
  protein VARCHAR(32),
  score FLOAT NOT NULL,
  score_std FLOAT,
  FOREIGN KEY (compartment) REFERENCES Compartment(code),
  FOREIGN KEY (protein) REFERENCES Protein(pid),
  PRIMARY KEY (protein, compartment)     
);
""",
"""
CREATE TABLE DataCoord (
  projection VARCHAR(64),
  protein VARCHAR(32),
  x FLOAT NOT NULL,
  y FLOAT NOT NULL,  
  FOREIGN KEY (projection) REFERENCES DataProjection(code),
  FOREIGN KEY (protein) REFERENCES Protein(pid),  
  PRIMARY KEY (protein, projection)     
);
""",
"""
CREATE TABLE ExpMember (
  exp_list VARCHAR(64),
  protein VARCHAR(32),
  score FLOAT,
  FOREIGN KEY (exp_list) REFERENCES ExpList(code),
  FOREIGN KEY (protein) REFERENCES Protein(pid),
  PRIMARY KEY (protein, exp_list)     
);
"""]
