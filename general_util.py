import os, sys, io, gzip
import numpy as np
from subprocess import Popen, PIPE
from constants import READ_BUFFER, UNIPROT_URL, UNIPROT_ALT_ID_URL

def info(msg):

  print('INFO: ' + msg)
  
 
def warn(msg):

  print('WARN: ' + msg)


def critical(msg):

  print('EXIT: ' + msg)
  print('STOP')
  sys.exit(0)
  
  
def open_file(file_path, mode=None, gzip_exts=('.gz','.gzip'), buffer_size=READ_BUFFER, partial=False, encoding='utf-8'):
  """
  GZIP agnostic file opening
  """

  if os.path.splitext(file_path)[1].lower() in gzip_exts:
    if mode and 'w' in mode:
      file_obj = io.BufferedWriter(gzip.open(file_path, mode), buffer_size)

    else:
      if partial:
        file_obj = io.BufferedReader(gzip.open(file_path, mode or 'rb'), buffer_size)

      else:
        try:
          file_obj = Popen(['zcat', file_path], stdout=PIPE).stdout
        except OSError:
          file_obj = io.BufferedReader(gzip.open(file_path, mode or 'rb'), buffer_size)

    file_obj = io.TextIOWrapper(file_obj, encoding=encoding)

  else:
    file_obj = open(file_path, mode or 'rU', buffer_size, encoding=encoding)

  return file_obj


def get_uniprot_alt_ids(alt_id_url=UNIPROT_ALT_ID_URL, cache_file='cache/uniprot_sec_acc.txt'):

  from  urllib import request

  info(f'Fetching UniProt secondary accessions')
  
  alt_id_dict = {}
  
  if not os.path.exists(cache_file):
    req = request.Request(alt_id_url)
 
    with request.urlopen(req) as f:
       response = f.read().decode('utf-8')
       
       info(f' .. writing {cache_file}')
       with open(cache_file, 'w') as file_obj:
         file_obj.write(response)
  
  with open(cache_file) as file_obj:
    lines = file_obj.readlines()
 
  for i, line in enumerate(lines):
    if line.startswith('Secondary AC'):
      break
  
  for line in lines[i+1:]:
    if line.strip():
      sec_id, prim_id = line.split()
      alt_id_dict[sec_id] = prim_id
  
  return alt_id_dict
  

def get_uniprot_columns(queries, columns=['id','protein_name','gene_primary'], batch_size=100):
  
  #from  urllib import request
  import requests
  from requests.adapters import HTTPAdapter, Retry

  #re_next_link = re.compile(r'<(.+)>; rel="next"')
  retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
  session = requests.Session()
  session.mount("https://", HTTPAdapter(max_retries=retries))

  if not isinstance(queries, (tuple, list)):
    queries = [queries]
  
  columns = list(columns)
  
  if len(queries) > batch_size:
    info(f'Fetching UniProt data for {len(queries):,} entries')

  a = 0
  n = len(queries)
  
  uniprot_dict = {}
  
  while a < n:
    b = min(n, a+batch_size)
    if len(queries) > batch_size:
      info(f' .. {a} - {b}', line_return=True)
    
    query_ids= '%20OR%20'.join(list(queries[a:b]))
    fields = '%2C'.join(['accession'] + columns)
    
    #req = request.Request(UNIPROT_URL.format(query_ids, fields))
    #with request.urlopen(req) as f:
    #   response = f.read()
 
    response = session.get(UNIPROT_URL.format(query_ids, fields)).text
 
    lines = response.split('\n')
    
    for line in lines[1:]:
      line = line.rstrip('\n')
 
      if not line:
        continue
        
      row = line.split('\t')  
      uniprot_dict[row[0]] = row[1:]
    
    a = b  
  
  if len(queries) > batch_size:
    info(f' .. {a}', line_return=True)
  
  if len(columns) == 1:
    for pid in uniprot_dict:
      uniprot_dict[pid] = uniprot_dict[pid][0]
  
  return uniprot_dict



def reformat_pd_data(in_pd_path, out_path, nfrac=10, start_col=17, idcol=3, step=1, sep='\t', nulls=('NAN', '', 'NA')):

  profile_dict = {}
  n_rec = 0 
  nulls = set(nulls)

  with open_file(in_pd_path) as file_obj, open_file(out_path, 'w') as out_file_obj:
    write = out_file_obj.write
    head1 = file_obj.readline()
 
    for line in file_obj:
      data = line.strip('\n').split(sep)
 
      if data:
        n_rec += 1
        pid = data[idcol]
        prof = data[start_col:start_col+step*nfrac:step]
        prof = ['nan' if v in nulls else v for v in prof]
        prof = '\t'.join(prof)
        out_line = f'{pid}\t{prof}\n'
        write(out_line)
        
  info(f'Wrote data for {nfrac} fractions covering {n_rec:,} proteins to {out_path}')
  
  
def plot_training_history(*histories, file_path=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    def _get_line(label, vals):
      texts = [label] + ['%.3e' % x for x in vals]
      return '\t'.join(texts) + '\n'
    
    cmap = cm.get_cmap('rainbow')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16,8)
    plot_options = {'linewidth':2, 'alpha':0.5} # for all charts
    
    color_dict = {'accuracy':'#FF2000','recall':'#BBBB00','precision':'#0080FF'}
    
    if file_path:
      table_file = os.path.splitext(file_path)[0] + '.tsv'
    else:
      table_file = 'training_history.tsv'
    
    with open(table_file, 'w') as out_file_obj:
      write = out_file_obj.write
      
      for i, history in enumerate(histories):
          if isinstance(history, dict):
            hd = history
          else:
            hd = history.history
            
          for j, metric in enumerate(hd):
            n = len(hd[metric])
            break
              
          epochs = np.arange(n) + 1
          m = len(hd)
          
          #plot_options['color'] = cmap(float(i % 10)/10)
 
          for j, metric in enumerate(hd):
            if 'loss' in metric:
              ax = ax1
            else:
              ax = ax2
                
            if 'val_' in metric:
              linestyle = '-'
              set_type = 'Test'
              met_name = metric[4:]
            else:
              linestyle = '--'
              set_type = 'Train'
              met_name = metric
            
            plot_options['color'] = cmap(float(j % m)/m)  # color_dict.get(met_name)

            if i > 0:
              label='%s %s %d' % (set_type, met_name, i)
            else:
              label='%s %s' % (set_type, met_name)
              
            ax.plot(epochs, hd[metric], label=label, linestyle=linestyle, **plot_options)
            write(_get_line(metric, hd[metric])) 
            
          ax1.set_title('Loss')
          ax1.set_xlabel('Iteration')
          ax2.set_title('Accuracy etc.')
          ax2.set_xlabel('Iteration')
          ax2.set_yticks(np.arange(0, 1.01, 0.05))
          ax2.set_xticks(np.arange(0, n, 10))

      ax1.legend()
      ax2.legend()
      
      ax1.grid(True, linewidth=0.5, alpha=0.5)
      ax2.grid(True, linewidth=0.5, alpha=0.5)
      
      if file_path:
        plt.savefig(file_path, dpi=300)
      else:
        plt.show()
 
 
def get_color_array(colors):
  
  n = len(colors)
  color_array = np.zeros((n, 3), float)
  
  for i, c in enumerate(colors):
    if isinstance(c, str) and c[0] == '#':
      red = int(c[1:3], 16) / 255.0
      grn = int(c[3:5], 16) / 255.0
      blu = int(c[5:7], 16) / 255.0
      color_array[i] = (red, grn, blu)

    else:
      red, grn, blu = colors[i]
      
      if isinstance(red, int):
        color_array[i] = (red/ 255.0, grn/ 255.0, blu/ 255.0)
  
  color_array = np.clip(color_array, 0.0, 1.0)
  
  return color_array
