package org.machine.learning.tensorface;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class TestClassify extends TestCase {

    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public TestClassify( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( TestClassify.class );
    }

    /**
     * Test
     */
    public void testClassify()
    {
        try {
            Classify c = new Classify();
            System.out.println("Error: " + c.classify("./model","./test") + "%");
            assertTrue( true );
        } catch (Exception ex) {
            System.err.println(ex);
            assertTrue( false );
        }
    }
}
