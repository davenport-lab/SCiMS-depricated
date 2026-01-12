[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## DEPRECATED

<img src="static/scims_logo.png" align=right width="25%">

<h1>
 <strong>SCiMS</strong>: <strong><u>S</u></strong>ex <strong><u>C</u></strong>alling from <strong><u>M</u></strong>etagenomic <strong><u>S</u></strong>equences   
 </h1>     

 
#### `SCiMS` is a tool for identifying the sex of a host organism based on the alignment of metagenomic sequences. 

----
## Installation 

`SCiMS` can be easily installed with the pip package manager

```
pip3 install git+https://github.com/Kobie-Kirven/SCIMS-V1.0
```
 
To confirm that the instillation was successful, run:
```
scims -h
```

## Usage
`SCiMS` can be used on any alignment data, regardless of the platform used for sequencing or the aligner that generated the alignment file. 

```
usage: scims [-h] [--i IDXTATS] [--s SCAFFOLD_IDS_FILE] [--X-id X_ID]
                   [--Y-id Y_ID]

Sex Assignment Script

options:
  -h, --help            show this help message and exit
  --i IDXTATS           idxstats file
  --s SCAFFOLD_IDS_FILE
                        File containing scaffold IDs of interest
  --X-id X_ID           Scaffold ID for X chromosome
  --Y-id Y_ID           Scaffold ID for Y chromosome
```


## Input Files
### Scaffolds.txt
Since most assemblies include scaffolds representing other DNA than simply genomic (ex. mitochondrial), it is necessary to define what scaffolds we are interested in using for our analysis. This can be specified with a ```scaffolds.txt`` file. This is a single-column text file where each row is a scaffold ID. Here is an example, 
```
NC_000001.11
NC_000002.12
NC_000003.12
NC_000004.12
NC_000005.10
NC_000006.12
NC_000007.14
NC_000008.11
...
``` 

### .idxstats files
A .idxstats file can easily be created with samtools. If you have a .bam file of interest, fun the following commands to generate the .idxstats file:

```shell
samtools index <bam_file>
```

```shell
samtools idxstats <bam_file> > <prefix>.idxstats
```

## Example run
Example files can be found in the ```test_data`` folder

```scims --i male.idxstats --s scafoolds.txt --X-id NC_000023.11 --Y-id NC_000024.10```

output:
```
=================================================

                                                  
  _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|  
_|        _|          _|    _|_|  _|_|  _|        
  _|_|    _|          _|    _|  _|  _|    _|_|    
      _|  _|          _|    _|      _|        _|  
_|_|_|      _|_|_|  _|_|_|  _|      _|  _|_|_|    

=================================================
Rx: 0.599
95% CI: 0.537 0.66
Ry: 0.517
95% CI: 0.464 0.57
```


