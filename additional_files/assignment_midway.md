## Task 1: Explore the data and write a brief summary

You will work with (part of) the public dataset PAD-UFES-20, you can download the data [here](https://data.mendeley.com/datasets/zr7vgbcyr2/1). Do not forget to cite the dataset in your report. 

Go through the data (images and meta-data) that you have available to understand what's available to you. For example:

* What types of diagnoses are there, and how do they relate to each other? You can use Google Scholar or other sources, to learn more background about these diagnoses. 
* Is there some missing data? Are there images of low quality? Etc.

Write a summary about your findings in a single groupXX_summary.md file:

* This can just be plain text document of 1-2 pages with text and some headings. Do not forget to add your sources/references. 
* Do not worry about the format too much, since later in the report course we will use LaTeX
* It is a requirement that each group member makes non-trivial updates to this file



## Task 2: Segment a set of images by hand

You will receive a list of images to segment as a group. You may discuss the segmentation process with group members before your start. Assign each group member an annotator ID, like a1-a5 if you are group A and you have 5 members. 

Create a file groupXX_annotation_comments.md, and add any notes there about your decisions, anything you find surprising and so on (this is optional). Do not add your names or other details in this file, but use your annotator ID.

All students must contribute to segmenting images. Create segmentations of these **by hand** with LabelStudio. You are allowed to use another program, but your annotation must follow the same format and be compatible with LabelStudio. 

If an image is too low quality, you can skip it, but try to have at least 100 annotated images in total. 

Place all your annotations into a separate groupXX_masks directory. The filenames must have the same format.

Create a file groupXX_imageids.csv where you keep track of which annotator (by their annotator ID) annotated the image, and that image's filename. 



## Hand-in mandatory assignment

You must hand in your group Github repository which contains:

* groupXX_summary.md with your findings about the data. Every group member must have made non-trivial commits to this file
* groupXX_annotation_comments.md with your findings about the annotation process 
* groupXX_imageids.csv with two columns: annotator ID, and filename of an annotation
* groupXX_masks/ directory, where all the annotations are stored. The annotations must be committed by the group members who made them

The hand-in is that you post the link to your repository on LearnIT.

The repository commits must show who has contributed to which file and when (that is, you shouldn't create a "clean" repository for the hand-in). 



