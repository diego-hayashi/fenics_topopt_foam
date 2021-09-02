#!/bin/bash

################################################################################
#                 🌀 Convert OpenFOAM parameters to Python 🌀                  #
################################################################################

#
# Copyright (C) 2020-2021 Diego Hayashi Alonso
#
# This file is part of FEniCS TopOpt Foam.
# 
# FEniCS TopOpt Foam is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FEniCS TopOpt Foam is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FEniCS TopOpt Foam. If not, see <https://www.gnu.org/licenses/>.
#

#
# Converts OpenFOAM parameters to Python using sed.
#
# Usage: ./openfoamtopy.sh arquivo
#		arquivo = filename
#
# Output: arquivo.py
#
# Returns:
#	0 (success)
# 	1 (error)
#
# Observation:
# 	The resulting Python file can be imported inside a Python code.
# 

# Usage
if [ $# -ne 1 ]
then
	echo "Usage: $0 filename"
	exit 1
fi

# Filename
arquivo="$1"
if [ ! -f "$arquivo" ]
then
	echo "Error: $arquivo does not exist"
	exit 1
fi

# Output file name
saida="$arquivo.py"

# File name for sed
arquivo_sed="$arquivo"

# Zeroth sed
arquivo_sed_temp="$arquivo.temp0"
sed "s/\t/ /g" "$arquivo_sed" >"$arquivo_sed_temp"
arquivo_sed="$arquivo_sed_temp"

# First sed
if grep --quiet "nonuniform List[^ ]* [1-9][0-9]*(.*);" "$arquivo_sed"
then
	arquivo_sed_temp="$arquivo.temp"
	sed "s/\(nonuniform List<vector>\) \([1-9][0-9]*\)(\(.*\));\$/\1\n\2\n(\n\3\n)\n;/;T;s/) *(/)\n(/g" "$arquivo_sed" |
	sed "s/\(nonuniform List<scalar>\) \([1-9][0-9]*\)(\(.*\));\$/\1\n\2\n(\n\3\n)\n;/;T;s/\([0-9]\) /\1\n/g" >"$arquivo_sed_temp"
	arquivo_sed="$arquivo_sed_temp"
fi
#
grep --quiet '^($' "$arquivo_sed"
if [ $? -ne 0 ] &&
     grep --quiet '^[0-9][0-9]*(.*)$' "$arquivo_sed"
then
	arquivo_sed_temp="$arquivo.temp2"
	sed -e 's/^\([0-9][0-9]*\)( */\1(\n/;
		s/  *)/)/g;
		T;s/ *\(([^)\n]*)\)/\1\n/g;tf;
		s/  */\n/g;s/)$/\n)/;
		:f;s/(\n/\n(\n/;
		' "$arquivo_sed" >"$arquivo_sed_temp"
	arquivo_sed="$arquivo_sed_temp"
fi

# Second sed
sed -n "
  :inicio;
  ### inserido no início
  1i import numpy as np
  1i data = {
  ### inserido no final
  \$a }
  # exclui comentários simplificado (qualquer linha inteira que tiver // antes de seção)
  /\/\//d;/\/\*/,/\*\//d;
  ### nome --> 'nome' : {
  s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\) *\$/\1'\2' : {/p;t;
  ### nome nonuniform List... --> 'nome' : np.array([
  ### <quantidade elementos> --> ignorado
  s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\)\( *\)nonuniform List.*/\1'\2'\3: np.array ([/p;Taf;n;
    :ai;n;
    :ai2;
    ### ) --> ], dtype = 'float'),
    s/^)\$/], dtype = 'float'),/p;t;
    ### ( --> ignorado
    s/^(\$/(/;tai;
    ### ignora eventual numero antes do ( na linha abaixo
    ### (valor valor ...) --> [valor, valor, ...],
    s/^[0-9]*(\(.*\)).*\$/[\1],/;Tam;s/ /, /g;p;bai;
    :am;
    ###valor --> valor,
    s/\$/,/p;bai;
    :af;
  ### nome nonuniform 0(); --> 'nome' : np.array([], dtype = 'float'),
  s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\)\( *\)nonuniform 0();.*/\1'\2'\3: np.array ([], dtype = 'float'),/p;t;
  ### nome uniform (valores) --> 'nome' : np.array([valores], dtype = 'float'),
  s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\)\( *\)uniform *(\([^)]*\).*/\1'\2'\3: np.array([\4], dtype = 'float'),/;Tbf;
    ### valores --> vírgula entre os valores
    s/\([0-9]\) /\1, /g;p;b;
    :bf;
    ## nome uniform valor --> 'nome' : np.array([valor], dtype = 'float'),
  s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\)\( *\)uniform *\([^;]*\).*/\1'\2'\3: np.array([\4], dtype = 'float'),/p;t;
  ## nome [valores] 1e-05; --> 'nome' : ([valores], 1e-05)
  ## nome [valores]; --> 'nome' : [valores],
  s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\)\( *\)\(\[.*\]\) *\([^;]*\);\$/\1'\2'\3: \4\5,/;Tcf;
    ## valores --> vírgula entre os valores
    s/\([0-9]\) /\1, /g;
    s/\(\[.*\]\)\(..*\),\$/(\1, \2),/;p;b;
    :cf;
  ### nome valor; --> 'nome' : 'valor',
  s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\)\( *\)\([^;]*\);\$/\1'\2'\3: '\4',/p;t;
  ### } --> },
  s/}/},/p;t;
  ### ignora linhas sem conteúdo ou apenas com {;
  s/^[ {;]*\$/{/;t;

  ### objeto implícito
  ### utiliza o nome 'substituir' para ser utilizado em outro sed
  s/^[0-9][0-9]*\$//;Tdf;n;
    s/^(\$//;Tdf;n;
    s/^\( *\)\([a-zA-Z][a-zA-Z0-9_]*\)\$/&/;Tdm;
      i 'substituir' : {
      binicio;
      :dm;
        s/^\([0-9][0-9]*\)(/\1(/;te0;
	i 'substituir' : np.array([
	bai2;
	:e0;
        i 'substituir' : [
	:ei;
        ### ) --> ]
        s/^)\$/]/p;t;
        ### ignora eventual numero antes do ( na linha abaixo
        ### (valor valor ...) --> [valor, valor, ...],
	s/^[0-9]*(\(.*\)).*\$/np.array([\1]),/;Tem;s/ /, /g;p;n;bei;
        :em;
        ###valor --> valor,
        s/\$/,/p;n;bei;
      :df;

  s/^)\$//;t;
  s/^/erro--> /p;
  " "$arquivo_sed" >"$saida"

# Errors
if [ $? -ne 0 ]; then exit 1; fi
grep --quiet erro "$saida"
if [ $? -eq 0 ]
then
	echo "File $saida contains errors"
	exit 1
fi

# Third sed
if grep --quiet "substituir" "$saida"
then
	# Searches for the name of the object to be substituted
	obj="$(sed -n "s/.*'object' *: *'\([^']*\).*/\1/p" "$saida")"
	sed --in-place "s/substituir/$obj/" "$saida"

	# Inserts '}' in the end of the file when the object is a structure
	if grep --quiet "'$obj' : {" "$saida"
	then
		sed --in-place '$a }' "$saida"
	fi

fi

# Substitute additional patterns

sed --in-place "
	s/'version'/'version'/;t;
	s/'List<word> 1(\(.*\))'/['\1']/g;
	s/'\([0-9\.+-][0-9\.eE+-]*\)'/\1/g;
	" "$saida"

case "$(basename "$arquivo")" in
	owner|neighbour)
		sed --in-place "s/dtype = 'float'/dtype = 'uint32'/" "$saida"
		;;
esac

if [ -f "$arquivo.temp0" ]; then rm -f "$arquivo.temp0"; fi
if [ -f "$arquivo.temp" ]; then rm -f "$arquivo.temp"; fi
if [ -f "$arquivo.temp2" ]; then rm -f "$arquivo.temp2"; fi

exit
