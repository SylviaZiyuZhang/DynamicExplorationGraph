package com.vc.deg;

import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;

import com.vc.deg.feature.FeatureFactory;

/**
 * The {@link FeatureSpace} knows how the data in the {@link FeatureVector} is structured
 * 
 * @author Nico Hezel
 */
public interface FeatureSpace {
	
	/**
	 * Supported matrices
	 * 
	 * @author Nico Hezel
	 */
	public static enum Metric {
		L2(1), InnerProduct(2), Manhatten(10);
		
		protected int id;
		
		Metric(int id) {
			this.id = id;
		}
		
		public int getId() {
			return id;
		}
	}
	
	/**
	 * Contains all registered {@link FeatureSpace} either via a service loader or manual
	 * 
	 * @author Nico Hezel
	 */
    public static class DefaultFeatureSpaceHolder {
    	
    	/**
    	 * Get all {@link FeatureSpace}s registered via a service loader
    	 * 
    	 * @return
    	 */
    	private static List<FeatureSpace> serviceLoaderFactories() {
    		final List<FeatureSpace> serviceLoaderFactories = new ArrayList<>();
    		for (FeatureSpace featureFactory : ServiceLoader.load(FeatureSpace.class))
    			serviceLoaderFactories.add(featureFactory);
    		return serviceLoaderFactories;
    	}
    	
    	/**
    	 * Register a new {@link FeatureSpace} manual
    	 * 
    	 * @param factory
    	 */
    	private static void registerFeatureSpace(FeatureSpace factory) {
    		if(registeredFactories.contains(factory) == false)
    			registeredFactories.add(factory);
    	}
    	
    	/**
    	 * Find a specific {@link FeatureSpace} based on the parameters
    	 * 
    	 * @param componentType
    	 * @param metric
    	 * @param isNative
    	 * @return
    	 */
    	private static FeatureSpace findFeatureSpace(String componentType, int metric, boolean isNative) {
    		
    		for (FeatureSpace registeredSpace : registeredFactories) 
    			if(componentType.equalsIgnoreCase(registeredSpace.getComponentType()) && metric == registeredSpace.metric() && isNative == registeredSpace.isNative())
    				return registeredSpace;
    		
    		// try again but this time non-native and potential slower feature spaces are allowed
	    	if(isNative) 
	    		for (FeatureSpace registeredSpace : registeredFactories) 
	    			if(componentType.equalsIgnoreCase(registeredSpace.getComponentType()) && metric == registeredSpace.metric())
	    				return registeredSpace;
    		
			return null;
    	}
    	
    	private final static List<FeatureSpace> registeredFactories = serviceLoaderFactories();
    }

    /**
	 * Register a new {@link FeatureFactory}
	 * 
	 * @param factory
	 */
    public static void registerFeatureSpace(FeatureSpace factory) {
    	DefaultFeatureSpaceHolder.registerFeatureSpace(factory);
    }
    
    /**
	 * Find a specific {@link FeatureFactory} based on the parameters
	 * Only used when loading a graph form drive and the correct {@link FeatureSpace} needs to be determined
	 * 
     * @param componentType
     * @param metric
     * @param isNative
     * @return
     */
    public static FeatureSpace findFeatureSpace(String componentType, int metric, boolean isNative) {
        return DefaultFeatureSpaceHolder.findFeatureSpace(componentType, metric, isNative);
    }
    

	/**
	 * Size in bytes per feature vector
	 * 
	 * @return
	 */
	public int featureSize();
	
	/**
	 * Dimensions of the feature
	 * 
	 * @return
	 */
	public int dims();
	
    /**
     * Feature data type, either one of the primitives or an object
     * 
     * @return
     */
    public String getComponentType();
	
	/**
	 * Identifier of the metric, either one of {@link Metric} or a custom one
	 * Number should be between 0 and 255. 
	 * 
	 * @return
	 */
	public int metric();
	
	/**
	 * If true expects all {@link FeatureVector}s to be in native memory.
	 * Native {@link FeatureSpace} can use the {@link FeatureVector#nativeAddress()} to access the raw data and compute distances in native code (e.g. LLVM, C).
	 * All none native {@link FeatureSpace} must use the read-methods of a FeatureVector to get the data.
	 * 
	 * @return
	 */
	public boolean isNative();
	
	/**
	 * This method does not check the size of the {@link FeatureVector} or if its in native memory or not.
	 * The selected {@link FeatureSpace} expects all those properties from every {@link FeatureSpace}.
	 * 
	 * @param f1
	 * @param f2
	 * @return
	 */
	public float computeDistance(FeatureVector f1, FeatureVector f2);
}