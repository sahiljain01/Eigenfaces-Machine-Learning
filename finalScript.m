% Need to install some video packages
% Takes the video input from a live stream

obj = videoinput('macvideo', 1);
set(obj, 'ReturnedColorSpace', 'RGB');
preview(obj); start(obj);
prompt3 = 'Are you ready to take your picture?';
x = input(prompt3, 's');
%
% % Takes a picture if the user says that they are ready to take a picture
%
if (strcmp(x, 'Yes'))
    A = getsnapshot(obj);
end


%  Restart from here until to
Save the image in case something goes wrong in the middle

imwrite(A, 'myimage.jpg');
img = imread('myimage.jpg');
% imshow(img)
A = img;

% Matlab Library to detect a face from an image
bbox = step(faceDetector, A);
I_faces = step(shapeInserter, A, int32(bbox));
imshow(I_faces), title('Detected faces');
disp(bbox)
I2 = imcrop(A,bbox(1, :));
imshow(I2)
I2 = rgb2gray(I2);
D = imresize(I2, [64 64]);


prompt2 = 'How many people would you like to compare against?';
y = input(prompt2);
leasterror = 100000000;
finalname = ' ';
for z = 1:y
    prompt3 = 'Who would you like to compare against?';
    name = input(prompt3, 's');
    imshow(C)
    error = func(name, D);
    if (error < leasterror)
        leasterror = error;
        finalname = name;
    end
end

disp(finalname);

function errorValue = func(x, imgCompare)
    directory = '~/Downloads/lfw1000';      % Full path of the directory to be searched in
    filesAndFolders = dir(directory);     % Returns all the files and folders in the directory
    filesInDir = filesAndFolders(~([filesAndFolders.isdir]));  % Returns only the files in the directory
    stringToBeFound = x;
    numOfFiles = length(filesInDir);
    realNum  = 1;
    i=1;

    while(i<=numOfFiles)
      filename = filesInDir(i).name;
      % Store the name of the file
      if (contains(filename, x))
      % Enter directory path here based on personal computer
          dirNew = '~/Downloads/lfw1000/';
        % Checks if the file contains the given filename substring
          dirNew = strcat(dirNew, filename);
          imgTemp = imread(dirNew);
          [rows, columns, numberOfColorChannels] = size(imgTemp);

         %check if the image is black and white or colored
         %if the image is colored, then it converts it to grayscale
          if numberOfColorChannels > 1
                imgTemp = rgb2gray(imgTemp);
          else

          end
          st.data{realNum} = double(imgTemp);
          realNum = realNum  + 1;
      end
      i = i+1;
    end

    realNum = realNum -1;

    %realNum is the number of elements in our training set
    % Find the average image in the training set
    avgImg = zeros(size(st.data{1}));
    scalar = 1/realNum;
    for k = 1:realNum
       avgImg = avgImg + (scalar*st.data{k});
    end

    %Normalizing each image by subtracting the mean of the training set
    %from each image

    for k = 1:realNum
     st.dataAvgI{k} = st.data{k} - avgImg;
    end

    %Creating a matrix 4096 by number of Images, where each column vector
    %is N*N representing one image

    AI = zeros(64*64, realNum);
    for k= 1:realNum
        AI(:, k) = st.dataAvgI{k}(:);
    end

    %Calculate the Covariance in AI, by finding the eigenvectors to AI*AI'
    %However, the calculation of eig vectors of 4096 * 4096 matrix is
    %intensive. By the theorem, where since (AA'*eig) = (k*eig),
    %(A*A'A*eig2) = (k2 * eig2), we can instead calculate the eigenvectors
    %of AI'*AI, and multiply those eigenvectors times AI to find the
    %k most important eigenvectors, where k is the number of images we
    %have.

    IATA = AI'*AI;

    %Orthogonal Diagonalization using svd

    [UI,SI,VI] = svd(IATA);

    %Importing the image that will be compared

    newImage = imgCompare;

   [rows, columns, numberOfColorChannels] = size(newImage);

         %check if the image is black and white or colored
         %if the image is colored, then it converts it to grayscale
      if numberOfColorChannels > 1
                newImage = rgb2gray(newImage);
      end

    newImage = double(newImage);

    %Normalize the imported image by subtracting the average image of the training set.
    newImageI = newImage - avgImg;

    projVectorI = zeros(realNum,1);

% Project the image onto the eigenspace (eigenface) that is spanned by
% k (number of images) eigenvectors given by the columns of SI in the svd
% of IATA

    totalImg = zeros(4096, 1);
    for k = 1:realNum
        vec = AI*VI(:, k);

        % Weight: projection of the image onto the eigenvectors
        % By intuition and as results show, the earlier eigenvectors are much
        % more important and as a result, the image is a better projection onto
        % the earlier eigenvectors and has greater weights.

        weight = vec'*newImageI(:);
        projVectorI(k,1)  = weight;
        totalImg = totalImg + (weight) * vec;
    end

    % totalImg is the reconstructed image

    totalImg = totalImg + reshape(avgImg, [4096 1]);

    %Error vector is defined by the difference between the reconstructed image
    % and the original image.

    errorVector = norm(totalImg - reshape(newImage, [4096 1]));
    errorValue = errorVector;

end
