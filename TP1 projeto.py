# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 18:21:28 2021

@author: leona
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import wavfile 
import huffmancodec as hf

P = [1,0,8,0,8,0,8,0,0,0,1]
a = [0,1,2,8]


############################################################FUNCOES############################################ 
def oco(P, a, flag): # funcao do histograma / P - fonte a ler, a - alfabeto, flag - decide o que vai ser devolvido pela funcao
    occ=[] # array das ocorrencias
    dic = {}
    for i in P: # poe o numero das ocorrencias num dicionario
        if i not in dic.keys():
            dic[i]=1
        else:
            dic[i]+=1
          
    for i in a: # constroi o array das ocorrencias pretendido
        if i not in dic.keys():
            occ += [0]
        else:
            occ+=[dic[i]]
            
    if flag == 1: # devolve dicionario sendo as chaves os elementos e os valores as respetivas ocorrencias
        return dic  

    occ=np.array(occ)   
    
    return occ # devolve array das respetivas ocorrencias

def histograma(P, a): # funcao do histograma / P - fonte a ler, a - alfabeto
    occ=oco(P,a,0) # devolve array das respetivas ocorrencias
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    a = np.array(a)
    
    nelem = a.shape[0]
    x = np.arange(0, nelem, 1,int)
    plt.xticks(x, a)
    ax.bar(x, occ)
    plt.show()

def entropia (oc, P): # funcao da entropia / oc - array de ocorrencias introduzido, P - respetiva fonte
    n_elem = len(P)
    
    if n_elem <= 1:
        return 0
    
    counts = oc
    probs = counts[np.nonzero(counts)] / n_elem # indices de todos os numeros != 0 
    n_prob = len(probs)
    
    if n_prob <= 1:
        return 0
    
    soma = 0
    for i in probs:
        soma -= i * np.log2(i)
    
    return soma # devolve valor da entropia
    
def reader(): # quarda num array apneas as letras do texto
    english = open('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/english.txt', 'r') 
    english_txt = english.read()
    eng = []

    for letra in english_txt:
        if letra.isalpha() == True:
            eng += [letra]
    english_array = np.asarray(eng)
    english.close()  
    return english_array # devolve o array

def bitmediosimb(oc, s, l): # recebe um dado dicionario de ocorrencias
    media_p = 0
    media_p_var = 0
    
    e=0
    for i in s: #range(len(l))
        media_p += (l[e] * oc[i]) #oc e dicionario
        media_p_var += ((l[e]**2) * oc[i])
        e+=1
              
    soma_oc=0   
    for i in oc.values():
        soma_oc += i
            
    media_p = media_p / soma_oc 
    
    media_p_var = media_p_var / soma_oc
    
    
    var = media_p_var - (media_p**2)
    print("var:")
    print(var)
      
    return media_p # devolve o valor dos bits medios por simbolo

def entropia_conj(P, a=None): # igual a entropia mas verifica o valor da entropia para elementos em conjunto 2 a 2

    fonte1 = []
    for i in range(0, len(P) - 1, 2):
        fonte1.append(str(P[i]) + "/" + str(P[i + 1]))
        
    a_conj = oco(fonte1,a,1)
            
    probabilidade = []

    for i in a_conj.values():
        probabilidade+=[i]
        
    #print(a_conj)    

    n_elem = len(fonte1)

    counts = np.array(probabilidade)
    probs = counts[np.nonzero(counts)] / n_elem # indices de todos os numeros != 0 
    n_prob = len(probs)
    
    if n_prob <= 1:
        return 0
    
    soma = 0
    for i in probs:
        soma -= i * np.log2(i)
    
    return soma 

def shazam(query, target, a, passo):
    
    k=0
    lista_inf_mutua=[]
    lista_c=[] # vai guardar o array janela sendo depois reiniciado
    query_oc = oco(query, a, 0)
    query_entropia = entropia (np.array(query_oc), np.array(query))
    
    while(k<=len(target) - len(query)):
        i = 0
        for i in range(len(query)):
            lista_c+=[query[i]]+[target[k+i]]
           
        target_oc = oco(target[k:len(query)+k], a, 0) 
        target_entropia = entropia (np.array(target_oc), np.array(target[k:len(query)+k]))
        
        inf_mutua = query_entropia + target_entropia - entropia_conj(lista_c, a)
        lista_inf_mutua+=[inf_mutua]
        k+=passo
        
        lista_c=[]
    return lista_inf_mutua

def grafico(P, a): # funcao do histograma / P - fonte a ler, a - alfabeto
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    a = np.array(a)
    nelem = a.shape[0]
    x = np.arange(0, nelem, 1,int)
    plt.xticks(x, a)
    ax.bar(x, P)
    plt.show()

#################################################################################################

#exercicio 3
print("Ex 3:")

img_kid = mpimg.imread('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/kid.bmp')
img_kid = img_kid.flatten()
img_homer = mpimg.imread('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/homer.bmp')
img_homer = img_homer.flatten()
img_homerBin = mpimg.imread('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/homerBin.bmp')
img_homerBin = img_homerBin.flatten()
[fs_guitarSolo, data_guitarSolo] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/guitarSolo.wav')

# kid:
histograma(img_kid, np.arange(256))
print("Entropia kid.bmp: ")
print(entropia(oco(img_kid, np.arange(256),0), img_kid))
# homer:    
histograma(img_homer, np.arange(256))
print("Entropia homer.bmp: ")
print(entropia(oco(img_homer, np.arange(256),0), img_homer))
# homerBin:
histograma(img_homerBin, np.arange(256))    
print("Entropia homerBin.bmp: ")
print(entropia(oco(img_homerBin, np.arange(256),0), img_homerBin))
# guitarSolo:
histograma(data_guitarSolo, np.arange(256))    
print("Entropia guitarSolo.wav: ")
print(entropia(oco(data_guitarSolo, np.arange(256),0), data_guitarSolo))
# english:
histograma(reader(), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
print("Entropia english.txt: ")
print(entropia(oco(reader(), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],0), reader()))

# exercicio 4

print("Ex 4:")

codec = hf.HuffmanCodec.from_data(img_kid)
s, l = codec.get_code_len()
oc_kid = oco(img_kid, np.arange(256), 1)   
  
codec1 = hf.HuffmanCodec.from_data(img_homer)
s1, l1 = codec1.get_code_len()
oc_homer = oco(img_homer, np.arange(256), 1)

codec2 = hf.HuffmanCodec.from_data(img_homerBin)
s2, l2 = codec2.get_code_len()
oc_homerBin = oco(img_homerBin, np.arange(256), 1)

codec3 = hf.HuffmanCodec.from_data(data_guitarSolo)
s3, l3 = codec3.get_code_len()    
oc_guitarSolo = oco(data_guitarSolo, np.arange(256), 1)

codec4 = hf.HuffmanCodec.from_data(reader())
s4, l4 = codec4.get_code_len()   
oc_english = oco(reader(), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], 1)

#kid:
media_p_kid = bitmediosimb(oc_kid, s, l)   
print("Bits médios por símbolo-> kid.bmp:", media_p_kid)

# homer:  
media_p_homer = bitmediosimb(oc_homer, s1, l1)
print("Bits médios por símbolo-> homer.bmp:", media_p_homer)

# homerBin:
media_p_homerBin = bitmediosimb(oc_homerBin, s2, l2)
print("Bits médios por símbolo-> homerBin.bmp:", media_p_homerBin)

# guitarSolo:
media_p_guitarSolo = bitmediosimb(oc_guitarSolo, s3, l3)
print("Bits médios por símbolo-> guitarSolo.wav:", media_p_guitarSolo)

# english:   

media_p_english = bitmediosimb(oc_english, s4, l4) 

print("Bits médios por símbolo-> english.txt:", media_p_english)

# exercicio 5

print("Ex 5:")

# kid:
    
print("Entropia conjunta kid.bmp: ")    
print(entropia_conj(img_kid, img_kid)/2) 

# homer:
    
print("Entropia conjunta homer.bmp: ")    
print(entropia_conj(img_homer, img_homer)/2)    
   
# homerBin:
    
print("Entropia conjunta homerBin.bmp: ")    
print(entropia_conj(img_homerBin, img_homerBin)/2)    

# guitarSolo:
    
print("Entropia conjunta guitarSolo.wav: ")     
print(entropia_conj(data_guitarSolo, data_guitarSolo)/2)    

# english:
    
print("Entropia conjunta english.txt: ")     
print(entropia_conj(reader(), ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])/2)    

# exercicio 6.a

print("Ex 6.a:")

query = [2, 6, 4, 10, 5, 9, 5, 8, 0, 8]

target = [6, 8, 9, 7, 2, 4, 9, 9, 4, 9, 1, 4, 8, 0, 1, 2, 2, 6, 3, 2, 0, 7, 4, 9, 5, 4, 8, 5, 2, 7, 8, 0, 7, 4, 8, 5, 7, 4, 3, 2, 2, 7, 3, 5, 2, 7, 4, 9, 9, 6]

alfabeto = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

passo = 1

print("Informacao mutua: ")
print(shazam(query, target, alfabeto, passo))

#exercicio 6.b

print("Ex 6.b:")

[fs_target01, data_target01] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/target01 - repeat.wav');
[fs_target02, data_target02] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/target02 - repeatNoise.wav');

passo = round(len(data_guitarSolo)/4)

print("Informacao mutua-> guitarSolo/target01: ")
print(shazam(data_guitarSolo, data_target01, np.arange(256), passo))
grafico(shazam(data_guitarSolo, data_target01, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_target01, np.arange(256), passo))))
print("Informacao mutua-> guitarSolo/target02: ")
print(shazam(data_guitarSolo, data_target02, np.arange(256), passo))
grafico(shazam(data_guitarSolo, data_target02, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_target02, np.arange(256), passo))))

#exercicio 6.c

print("Ex 6.c:")

[fs_song01, data_song01] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/Song01.wav');
[fs_song02, data_song02] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/Song02.wav');
[fs_song03, data_song03] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/Song03.wav');
[fs_song04, data_song04] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/Song04.wav');
[fs_song05, data_song05] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/Song05.wav');
[fs_song06, data_song06] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/Song06.wav');
[fs_song07, data_song07] = wavfile.read('C:/Users/leona/OneDrive/Ambiente de Trabalho/TI (2 ano)/TP1 data/Song07.wav');

lista_decrescente = []

print("Informacao mutua maxima de song01: ")
max1 = max(shazam(data_guitarSolo, data_song01, np.arange(256), passo))
print(max1)
grafico(shazam(data_guitarSolo, data_song01, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_song01, np.arange(256), passo))))
lista_decrescente+=[max1]

print("Informacao mutua maxima de song02: ")
max2 = max(shazam(data_guitarSolo, data_song02, np.arange(256), passo))
print(max2)
grafico(shazam(data_guitarSolo, data_song02, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_song02, np.arange(256), passo))))
lista_decrescente+=[max2]

print("Informacao mutua maxima de song03: ")
max3 = max(shazam(data_guitarSolo, data_song03, np.arange(256), passo))
print(max3)
grafico(shazam(data_guitarSolo, data_song03, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_song03, np.arange(256), passo))))
lista_decrescente+=[max3]

print("Informacao mutua maxima de song04: ")
max4 = max(shazam(data_guitarSolo, data_song04, np.arange(256), passo))
print(max4)
grafico(shazam(data_guitarSolo, data_song04, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_song04, np.arange(256), passo))))
lista_decrescente+=[max4]

print("Informacao mutua maxima de song05: ")
max5 = max(shazam(data_guitarSolo, data_song05, np.arange(256), passo))
print(max5)
grafico(shazam(data_guitarSolo, data_song05, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_song05, np.arange(256), passo))))
lista_decrescente+=[max5]

print("Informacao mutua maxima de song06: ")
max6 = max(shazam(data_guitarSolo, data_song06, np.arange(256), passo))
print(max6)
grafico(shazam(data_guitarSolo, data_song06, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_song06, np.arange(256), passo))))
lista_decrescente+=[max6]

print("Informacao mutua maxima de song07: ")
max7 = max(shazam(data_guitarSolo, data_song07, np.arange(256), passo))
print(max7)
grafico(shazam(data_guitarSolo, data_song07, np.arange(256), passo), np.arange(len(shazam(data_guitarSolo, data_song07, np.arange(256), passo))))
lista_decrescente+=[max7]

lista_decrescente.sort(reverse = True)
print(lista_decrescente)