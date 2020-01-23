// include the header of your analysis task here! for classes already compiled by aliBuild,
// precompiled header files (with extension pcm) are available, so that you do not need to
// specify includes for those. for your own task however, you (probably) have not generated a
// pcm file, so we need to include it explicitly
// #include "AliAnalysisTaskMyTask.h"

void runAnalysis()
{
    // set if you want to run the analysis locally (kTRUE), or on grid (kFALSE)
    Bool_t local = kTRUE;
    // if you run on grid, specify test mode (kTRUE) or full grid model (kFALSE)
    Bool_t gridTest = kTRUE;

    Bool_t isMC = kTRUE;

    // since we will compile a class, tell root where to look for headers
#if !defined (__CINT__) || defined (__CLING__)
    // ROOT6
    gInterpreter->ProcessLine(".include $ROOTSYS/include");
    gInterpreter->ProcessLine(".include $ALICE_ROOT/include");
#else
    // ROOT5
    gROOT->ProcessLine(".include $ROOTSYS/include");
    gROOT->ProcessLine(".include $ALICE_ROOT/include");
#endif

    // create the analysis manager
    AliAnalysisManager *mgr = new AliAnalysisManager("AnalysisTaskExample");
    AliAODInputHandler *aodH = new AliAODInputHandler();
    mgr->SetInputEventHandler(aodH);


    // compile the class and load the add task macro
    // here we have to differentiate between using the just-in-time compiler
    // from root6, or the interpreter of root5
#if !defined (__CINT__) || defined (__CLING__)
    gInterpreter->LoadMacro("AliAnalysisTaskJetExtractor.cxx++g");

    if(isMC) AliPhysicsSelectionTask* physicsSelectionTask = reinterpret_cast<AliPhysicsSelectionTask*>(gInterpreter->ExecuteMacro("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C(kTRUE, kTRUE)"));
    else     AliPhysicsSelectionTask* physicsSelectionTask = reinterpret_cast<AliPhysicsSelectionTask*>(gInterpreter->ExecuteMacro("$ALICE_PHYSICS/OADB/macros/AddTaskPhysicsSelection.C(kFALSE, kTRUE)"));

    AliEmcalJetTask* emcalJetTask                 = reinterpret_cast<AliEmcalJetTask*>(        gInterpreter->ExecuteMacro("$ALICE_PHYSICS/PWGJE/EMCALJetTasks/macros/AddTaskEmcalJet.C(\"\", \"\", AliJetContainer::antikt_algorithm, 0.4, AliJetContainer::kChargedJet, 0.15, 0.3, 0.005, AliJetContainer::E_scheme, \"Jet\", 0., kTRUE, kFALSE)"));

    emcalJetTask->SetNeedEmcalGeom(kFALSE);
    emcalJetTask->SelectCollisionCandidates(AliVEvent::kINT7);

    AliTrackContainer* trackCont = new AliTrackContainer("tracks");
    trackCont->SetFilterHybridTracks(kTRUE);
    trackCont->SetParticlePtCut(0.15);
    trackCont->SetTrackFilterType(AliEmcalTrackSelection::kCustomTrackFilter);
    trackCont->SetAODFilterBits((1<<4)|(1<<9));
    emcalJetTask->AdoptParticleContainer(trackCont);

    if (isMC){
        AliAnalysisTaskJetExtractor *task_b     = reinterpret_cast<AliAnalysisTaskJetExtractor*>(gInterpreter->ExecuteMacro("AddTaskJetExtractor.C(\"tracks\", \"\", \"Jet_AKTChargedR040_tracks_pT0150_E_scheme\", \"\", 0.4, \"bJets\")"));
        AliAnalysisTaskJetExtractor *task_c     = reinterpret_cast<AliAnalysisTaskJetExtractor*>(gInterpreter->ExecuteMacro("AddTaskJetExtractor.C(\"tracks\", \"\", \"Jet_AKTChargedR040_tracks_pT0150_E_scheme\", \"\", 0.4, \"cJets\")"));
        AliAnalysisTaskJetExtractor *task_light = reinterpret_cast<AliAnalysisTaskJetExtractor*>(gInterpreter->ExecuteMacro("AddTaskJetExtractor.C(\"tracks\", \"\", \"Jet_AKTChargedR040_tracks_pT0150_E_scheme\", \"\", 0.4, \"udsgJets\")"));
    }
    else{
        AliAnalysisTaskJetExtractor *task = reinterpret_cast<AliAnalysisTaskJetExtractor*>(gInterpreter->ExecuteMacro("AddTaskJetExtractor.C(\"tracks\", \"\", \"Jet_AKTChargedR040_tracks_pT0150_E_scheme\", \"\", 0.4, \"allJets\")"));
    }


#else
    // gROOT->LoadMacro("AliAnalysisTaskJetExtractor.cxx++g");
    // gROOT->LoadMacro("AddTaskJetExtractor.C");
    // AliAnalysisTaskJetExtractor *task = AddTaskJetExtractor();
#endif


    if(!mgr->InitAnalysis()) return;
    mgr->SetDebugLevel(0);
    mgr->PrintStatus();
    mgr->SetUseProgressBar(1, 25);

/************************
    globalvariables.C

    TString kJetAkTName(Form("Jet_AKTChargedR040_PicoTracks_pT0150_pt_scheme"));
    TString kJetRhokTName(Form("Jet_KTChargedR040_PicoTracks_pT0150_pt_scheme"));
    Double_t kAreaCut = 0.;
    Double_t kJetLeadingTrackBias = 0;
    Double_t kClusPtCut = 0.30;
    Double_t kTrackPtCut = 0.15;
    Double_t kPartLevPtCut = 0;
    Double_t kGhostArea = 0.01;
    Double_t kMaxTrackPt = 1000;
    Double_t kJetPtCut = 1;
    //boost::v1_53_0 cgal::v4.4 fastjet::v3.0.6_1.012
    ////// OTHER EMCAL RELATED VARIABLES ////
    const char* kDataPeriod="LHC12c";
    TString kMatchingChainStr="16:512:includeNoITS=kFALSE doProp=kFALSE doAttemptProp=kTRUE isMC=kFALSE";
    TString kClusterName="EmcCaloClusters";
    TString kRhoTaskName="ExternalRhoTask";
    TString kTracksName="PicoTracks";
*************************/

    if(local) {
        // if you want to run locally, we need to define some input
        TChain* chain = new TChain("aodTree");
        // add a few files to the chain (change this so that your local files are added)
        //chain->Add("AliAOD.root");
        chain->Add("sim/2016/LHC16h3/17/244540/AOD/AliAOD.root");

        // start the analysis locally, reading the events from the tchain
        mgr->StartAnalysis("local", chain);
    } else {
        // if we want to run on grid, we create and configure the plugin
        AliAnalysisAlien *alienHandler = new AliAnalysisAlien();
        // also specify the include (header) paths on grid
        alienHandler->AddIncludePath("-I. -I$ROOTSYS/include -I$ALICE_ROOT -I$ALICE_ROOT/include -I$ALICE_PHYSICS/include");
        // make sure your source files get copied to grid
        // alienHandler->SetAdditionalLibs("AliAnalysisTaskMyTask.cxx AliAnalysisTaskMyTask.h");
        // alienHandler->SetAnalysisSource("AliAnalysisTaskMyTask.cxx");
        // select the aliphysics version. all other packages
        // are LOADED AUTOMATICALLY!
        alienHandler->SetAliPhysicsVersion("vAN-20191002_ROOT6-1");

        // set the Alien API version
        alienHandler->SetAPIVersion("V1.1x");

        // define the output folders
        alienHandler->SetGridWorkingDir("myWorkingDir_LHC16h3/ptbin10"); // ### !!!
        alienHandler->SetGridOutputDir("myOutputDir");

        // select the input data
        if(isMC){
            // path = /alice/sim/2016/LHC16h3/10/244480/AOD/088/AliAOD.root
            alienHandler->SetGridDataDir("/alice/sim/2016/LHC16h3/10/");
            alienHandler->SetDataPattern("*AOD/*AliAOD.root");
            alienHandler->SetRunPrefix("");  // MC has no prefix, data has prefix 000
        }
        else{
            // path = /alice/data/2015/LHC15n/000244453/pass4/AOD208/0090/AliAOD.root
            alienHandler->SetGridDataDir("/alice/data/2015/LHC15n/");
            alienHandler->SetDataPattern("*pass4/AOD208*AliAOD.root");
            alienHandler->SetRunPrefix("000");  // MC has no prefix, data has prefix 000
        }
        // runnumbers
        // based on pass3 !!!
        alienHandler->AddRunNumber(244628);
        alienHandler->AddRunNumber(244627);
        alienHandler->AddRunNumber(244626);
        alienHandler->AddRunNumber(244619);
        alienHandler->AddRunNumber(244618);
        alienHandler->AddRunNumber(244617);
        alienHandler->AddRunNumber(244542);
        alienHandler->AddRunNumber(244540);
        alienHandler->AddRunNumber(244531);
        alienHandler->AddRunNumber(244484);
        alienHandler->AddRunNumber(244483);
        alienHandler->AddRunNumber(244482);
        alienHandler->AddRunNumber(244481);
        alienHandler->AddRunNumber(244480);
        alienHandler->AddRunNumber(244456);
        alienHandler->AddRunNumber(244453);
        alienHandler->AddRunNumber(244421);
        alienHandler->AddRunNumber(244416);
        alienHandler->AddRunNumber(244377);
        alienHandler->AddRunNumber(244364);
        alienHandler->AddRunNumber(244359);
        alienHandler->AddRunNumber(244355);
        alienHandler->AddRunNumber(244351);
        //alienHandler->AddRunNumber(244343); // not in RunList_*_calo
        alienHandler->AddRunNumber(244340);

        // number of files per subjob
        alienHandler->SetSplitMaxInputFileNumber(3);
        alienHandler->SetExecutable("myTask.sh");
        // specify how many seconds your job may take
        alienHandler->SetTTL(54000); // max 86000
        alienHandler->SetJDLName("myTask.jdl");

        alienHandler->SetOutputToRunNo(kTRUE);
        alienHandler->SetKeepLogs(kTRUE);

        alienHandler->SetCheckCopy(kFALSE); // ### !!!
        // merging: run with kTRUE to merge on grid
        // after re-running the jobs in SetRunMode("terminate")
        // (see below) mode, set SetMergeViaJDL(kFALSE)
        // to collect final results
        alienHandler->SetMaxMergeStages(3);
        alienHandler->SetMergeViaJDL(kTRUE);  // ### !!! first kTRUE, then kFALSE for the last step of merging

        // connect the alien plugin to the manager
        mgr->SetGridHandler(alienHandler);
        if(gridTest) {
            // speficy on how many files you want to run
            alienHandler->SetNtestFiles(1);
            // and launch the analysis
            alienHandler->SetRunMode("test");
            mgr->StartAnalysis("grid");
        } else {
            // else launch the full grid analysis
            alienHandler->SetRunMode("full");  // ### !!! "full", then "terminate" for first step(s) of merging
            mgr->StartAnalysis("grid");
        }
    }
}
