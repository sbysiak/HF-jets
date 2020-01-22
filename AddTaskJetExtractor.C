AliAnalysisTaskJetExtractor* AddTaskJetExtractor (TString trackArray="tracks", TString clusterArray="", TString jetArray="Jet_AKTChargedR040_tracks_pT0150_E_scheme", TString rhoObject="", Double_t jetRadius=0.4, const char* taskNameSuffix="allJets")
{
    cout << "\n\n\n\n\n\n\n MY ADD TASK !!!!! \n\n\n\n\n\n\n";

    TString mode = TString(taskNameSuffix);

    // tasks JetExt depends on


    /////// Jet Extractor Task ///
    AliRDHFJetsCutsVertex* cuts = new AliRDHFJetsCutsVertex("jetCuts");
    AliAnalysisTaskJetExtractor* myTask = AliAnalysisTaskJetExtractor::AddTaskJetExtractor(trackArray, clusterArray, jetArray, rhoObject, jetRadius, cuts, taskNameSuffix);

    myTask->SetForceBeamType(AliAnalysisTaskEmcal::kpp);
    myTask->SetVzRange(-10,10);
    myTask->SetUseBuiltinEventSelection(kFALSE);

    myTask->SetSaveConstituents(1);
    myTask->SetSaveConstituentsIP(1);
    myTask->SetSaveConstituentPID(1);
    myTask->SetSaveJetShapes(1);
    myTask->SetSaveJetSplittings(1);
    myTask->SetSaveSecondaryVertices(1);
    myTask->SetSaveTriggerTracks(1);
    myTask->SetSaveMCInformation(1);

    myTask->GetJetContainer(0)->SetJetRadius(jetRadius);
    myTask->GetJetContainer(0)->SetPercAreaCut(0.557);
    myTask->GetJetContainer(0)->SetJetEtaLimits(-(0.9-jetRadius), (0.9-jetRadius));

    myTask->GetTrackContainer(0)->SetTrackFilterType(AliEmcalTrackSelection::kCustomTrackFilter);
    myTask->GetTrackContainer(0)->SetAODFilterBits((1<<4)|(1<<9));


    if (mode.EqualTo("allJets")){
        myTask->GetJetTree()->AddExtractionPercentage(0,5   , 0.1);
        myTask->GetJetTree()->AddExtractionPercentage(5,10  , 0.3);
        myTask->GetJetTree()->AddExtractionPercentage(10,20 , 0.6);
        myTask->GetJetTree()->AddExtractionPercentage(20,40 , 1.0);
        myTask->GetJetTree()->AddExtractionPercentage(40,200, 1.0);
        // myTask->GetJetTree()->AddExtractionPercentage(0,200, 1.0);
        return myTask;
    }
    

    myTask->SetIsPythia(kTRUE);
    myTask->SetNumberOfPtHardBins(21);
    Int_t arr[22] = {0,5,7,9,12,16,21,28,36,45,57,70,85,99,115,132,150,169,190,212,235,1000000};
    TArrayI bins(22, arr);
    myTask->SetUserPtHardBinning(bins);

    if (mode.EqualTo("bJets")){
        myTask->GetJetTree()->AddExtractionJetTypeHM(5);
        myTask->GetJetTree()->AddExtractionPercentage(0,5   , 0.1);
        myTask->GetJetTree()->AddExtractionPercentage(5,10  , 0.3);
        myTask->GetJetTree()->AddExtractionPercentage(10,20 , 0.6);
        myTask->GetJetTree()->AddExtractionPercentage(20,40 , 1.0);
        myTask->GetJetTree()->AddExtractionPercentage(40,200, 1.0);
        // myTask->GetJetTree()->AddExtractionPercentage(0,200, 1.0);
        return myTask;
    }
    if (mode.EqualTo("cJets")){ // 2x less than b except for last
        myTask->GetJetTree()->AddExtractionJetTypeHM(4);
        myTask->GetJetTree()->AddExtractionPercentage(0,5   , 0.05);
        myTask->GetJetTree()->AddExtractionPercentage(5,10  , 0.15);
        myTask->GetJetTree()->AddExtractionPercentage(10,20 , 0.3);
        myTask->GetJetTree()->AddExtractionPercentage(20,40 , 0.5);
        myTask->GetJetTree()->AddExtractionPercentage(40,200, 1.0);
        // myTask->GetJetTree()->AddExtractionPercentage(0,200, 1.0);
        return myTask;
    }
    if (mode.EqualTo("udsgJets")){ // 10x less then b except for last
        myTask->GetJetTree()->AddExtractionJetTypeHM(1);
        myTask->GetJetTree()->AddExtractionJetTypeHM(3);
        myTask->GetJetTree()->AddExtractionPercentage(0,5   , 0.0005);
        myTask->GetJetTree()->AddExtractionPercentage(5,10  , 0.03);
        myTask->GetJetTree()->AddExtractionPercentage(10,20 , 0.1); // +
        myTask->GetJetTree()->AddExtractionPercentage(20,40 , 0.2);
        myTask->GetJetTree()->AddExtractionPercentage(40,80 , 0.6);
        myTask->GetJetTree()->AddExtractionPercentage(80,200, 1.0);
        // myTask->GetJetTree()->AddExtractionPercentage(0,200, 1.0);
        return myTask;
    }

}
