package it.unibz.krdb.obda.protege4.gui.action;

/*
 * #%L
 * ontop-protege4
 * %%
 * Copyright (C) 2009 - 2013 KRDB Research Centre. Free University of Bozen Bolzano.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */

import it.unibz.krdb.obda.model.OBDAModel;
import it.unibz.krdb.obda.model.impl.OBDAModelImpl;
import it.unibz.krdb.obda.protege4.core.OBDAModelManager;
import it.unibz.krdb.obda.protege4.panels.OBDAModelStatisticsPanel;
import it.unibz.krdb.obda.protege4.utils.DialogUtils;
import it.unibz.krdb.obda.utils.VirtualABoxStatistics;

import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JPanel;

import org.protege.editor.core.ui.action.ProtegeAction;
import org.protege.editor.owl.OWLEditorKit;

public class ABoxStatisticsAction extends ProtegeAction {

	private static final long serialVersionUID = 3322509244957306932L;

	private OWLEditorKit editorKit = null;
	private OBDAModel obdaModel = null;
	private VirtualABoxStatistics statistics = null;
		
	@Override
	public void initialise() throws Exception {
		editorKit = (OWLEditorKit)getEditorKit();		
		obdaModel = ((OBDAModelManager)editorKit.get(OBDAModelImpl.class.getName())).getActiveOBDAModel();
		statistics = new VirtualABoxStatistics(obdaModel);
	}

	@Override
	public void dispose() throws Exception {
		// Does nothing.
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		statistics.refresh();  // refresh the statistics every time users click the menu.
		
		JDialog dialog = new JDialog();
		dialog.setModal(true);
		dialog.setSize(520, 400);
		dialog.setLocationRelativeTo(null);
		dialog.setTitle("OBDA Model Statistics");
		
		OBDAModelStatisticsPanel pnlStatistics = new OBDAModelStatisticsPanel(statistics);
		JPanel pnlCommandButton = createButtonPanel(dialog); 
		dialog.setLayout(new BorderLayout());
		dialog.add(pnlStatistics, BorderLayout.CENTER);
		dialog.add(pnlCommandButton, BorderLayout.SOUTH);
		
		DialogUtils.installEscapeCloseOperation(dialog);
		
		dialog.pack();
		dialog.setVisible(true);
		
	}
	
	private JPanel createButtonPanel(final JDialog parent) {
		JPanel panel = new JPanel();
		panel.setLayout(new FlowLayout());
		
		JButton cmdCloseInformation = new JButton();
		cmdCloseInformation.setText("Close Information");
		cmdCloseInformation.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                parent.setVisible(false);
                parent.removeAll();
                parent.dispose();
            }
        });		
		panel.add(cmdCloseInformation);
		
		return panel;
	}
}
