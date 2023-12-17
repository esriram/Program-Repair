package it.unibz.krdb.sql;

import it.unibz.krdb.obda.io.ModelIOManager;
import it.unibz.krdb.obda.model.OBDADataFactory;
import it.unibz.krdb.obda.model.OBDAModel;
import it.unibz.krdb.obda.model.impl.OBDADataFactoryImpl;
import it.unibz.krdb.obda.owlrefplatform.core.QuestConstants;
import it.unibz.krdb.obda.owlrefplatform.core.QuestPreferences;
import it.unibz.krdb.obda.owlrefplatform.owlapi3.*;
import org.junit.Test;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.OWLObject;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.reasoner.SimpleConfiguration;
import java.io.File;

public class SAPHANATest {

    final String owlfile = "src/test/resources/example/SAPbooktutorial.owl";
    final String obdafile = "src/test/resources/example/SAPbooktutorial.obda";

//    final String owlfile = "src/test/resources/example/exampleBooks.owl";
//    final String obdafile = "src/test/resources/example/exampleBooks.obda";

    @Test
    public void runQuery() throws Exception {
            /* 
             * Load the ontology from an external .owl file. 
            */
        //Class.forName("com.sap.db.jdbc.Driver");
        OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
        OWLOntology ontology = manager.loadOntologyFromOntologyDocument(new File(owlfile));

            /* 
            * Load the OBDA model from an external .obda file 
            */
        OBDADataFactory fac = OBDADataFactoryImpl.getInstance();
        OBDAModel obdaModel = fac.getOBDAModel();
        ModelIOManager ioManager = new ModelIOManager(obdaModel);
        ioManager.load(obdafile);

            /* 
            * * Prepare the configuration for the Quest instance. The example below shows the setup for 
            * * "Virtual ABox" mode 
            */
        QuestPreferences preference = new QuestPreferences();
        preference.setCurrentValueOf(QuestPreferences.ABOX_MODE, QuestConstants.VIRTUAL);

            /* 
            * Create the instance of Quest OWL reasoner. 
            */
        QuestOWLFactory factory = new QuestOWLFactory();
        factory.setOBDAController(obdaModel);
        factory.setPreferenceHolder(preference);

        QuestOWL reasoner = (QuestOWL) factory.createReasoner(ontology, new SimpleConfiguration());

            /* 
            * Prepare the data connection for querying. 
            */
        QuestOWLConnection conn = reasoner.getConnection();
        QuestOWLStatement st = conn.createStatement();

            /* 
            * Get the book information that is stored in the database 
            */
        String sparqlQuery =
                "PREFIX : <http://meraka/moss/exampleBooks.owl#>\n" +
                        "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
                        "select ?x ?y where {?x rdf:type :Author. ?x :name ?y.} limit 5 offset 2";
        try {
            long t1 = System.currentTimeMillis();
            QuestOWLResultSet rs = st.executeTuple(sparqlQuery);
            int columnSize = rs.getColumnCount();
            while (rs.nextRow()) {
                for (int idx = 1; idx <= columnSize; idx++) {
                    OWLObject binding = rs.getOWLObject(idx);
                    System.out.print(binding.toString() + ", ");
                }
                System.out.print("\n");
            }
            rs.close();
            long t2 = System.currentTimeMillis();

                /* 
                * Print the query summary 
                */
            QuestOWLStatement qst = (QuestOWLStatement) st;
            String sqlQuery = qst.getUnfolding(sparqlQuery);
            System.out.println();
            System.out.println("The input SPARQL query:");
            System.out.println("=======================");
            System.out.println(sparqlQuery);
            System.out.println();
            System.out.println("The output SQL query:");
            System.out.println("=====================");
            System.out.println(sqlQuery);
            System.out.println("Query Execution Time:");
            System.out.println("=====================");
            System.out.println((t2-t1) + "ms");

        } finally {
            /* 
            * Close connection and resources 
            * */
            if (st != null && !st.isClosed()) {
                st.close();
            }

            if (conn != null && !conn.isClosed()) {
                conn.close();
            }
            reasoner.dispose();
        }
    }
    /** 
     * Main client program 
     * */
    public static void main(String[] args) {
        try {
            SAPHANATest example = new SAPHANATest();
            example.runQuery();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
