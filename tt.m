% abre arquvivo de texto
file = fopen('histTumorT9.txt','wt')
% loop nas imagens, no meu caso de 0 a 49
for a = 0:49
   % imagens com nome 0.jpg, 1.jpg...
   filename = ['TWT_cinza/' num2str(a) '.jpg'];
   img = imread(filename);
   % o segundo parametro voce pode mudar, no meu caso 3 5 7 9 deu bom
   LPQhist = lpqNEW(img,9,1,1,'nh');
   
   fprintf(file,'%f \n', LPQhist);

end
fclose(file);
