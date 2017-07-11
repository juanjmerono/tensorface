package org.machine.learning.tensorface;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Classify {
	
    private byte[] graphDef;
    private List<String> labels;

    private int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }
    
    private byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }

    private List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }
    
    private byte[] getFileImage(File file) throws IOException {
        InputStream is = new FileInputStream(file);
        ByteArrayOutputStream os = new ByteArrayOutputStream();

        byte[] b = new byte[2048];
        int length;

        while ((length = is.read(b)) != -1) {
            os.write(b, 0, length);
        }

        is.close();
        os.close();
        return os.toByteArray();
    }
    
    private FilenameFilter getFilter(boolean images) {
    	FilenameFilter folder = new FilenameFilter(){
    		@Override
    		public boolean accept(File dir, String name) {
    			return new File(dir,name).isDirectory();
    		}
    	};
    	FilenameFilter image = new FilenameFilter(){
    		@Override
    		public boolean accept(File dir, String name) {
    			return name.toLowerCase().endsWith(".jpg");
    		}
    	};
    	return images?image:folder;
    }

    public int classify(String modelFolder, String datasetFolder) throws IOException {
        graphDef = readAllBytesOrExit(Paths.get(modelFolder, "retrained_graph.pb"));
        labels = readAllLinesOrExit(Paths.get(modelFolder, "retrained_labels.txt"));
        File testDataset = new File(datasetFolder);
        System.out.println(testDataset.getAbsolutePath());
        int errors = 0; int tests = 0;
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g)) {
            	for (String folder : testDataset.list(getFilter(false))) {
            		File classFolder = new File(testDataset,folder);
            		for (String file: classFolder.list(getFilter(true))) {
            			Tensor image = Tensor.create(getFileImage(new File(classFolder,file)));
            			Tensor result = s.runner()
	            			.feed("DecodeJpeg/contents", image)
	            			.fetch("final_result")
	            			.run().get(0);
		                final long[] rshape = result.shape();
		                /*if (result.numDimensions() != 2 || rshape[0] != 1) {
		                    throw new RuntimeException(
		                            String.format(
		                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
		                                    Arrays.toString(rshape)));
		                }*/
		                int nlabels = (int) rshape[1];
		                float[] labelProbabilities = result.copyTo(new float[1][nlabels])[0];
		                int bestLabelIdx = maxIndex(labelProbabilities);
		                if (!labels.get(bestLabelIdx).equals(folder)) {
		                	errors++;
			                System.out.println(
			                        String.format(
			                                "BEST MATCH[%s/%s]: %s (%.2f%% likely)", folder, file,
			                                labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
		                }
		                tests++;
            		}
            	}
            	System.out.println(String.format("Total: %d, errors %d (%.2f%%)", tests, errors, (float)errors/tests));
            }
        }
        return Math.round((errors/tests)*100);
    }

	public static void main(String[] args) throws IOException {
		Classify c = new Classify();
        c.classify(args[0],args[1]);
	}

}
