// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/Common/interface/Ptr.h>
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
//ROOT includes
#include "TTree.h"
#include <TFile.h>
#include <TROOT.h>
#include "TBranch.h"
#include <string>
#include <vector>
#include "TSystem.h"

//CMSSW includes
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

class DeepNtuplizer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit DeepNtuplizer(const edm::ParameterSet&);
  ~DeepNtuplizer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  // ----------member data --------------------------- 
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<pat::JetCollection>     jetToken_;

  TFile *file_ = new TFile("output.root","recreate");
  TTree *tree_ = new TTree("tree","tree");

  // labels (MC truth)
  float gen_pt_;

  // global event variables
  unsigned int npv_;

  // jet variables
  float jet_pt_;
  float  jet_eta_;

  // PF candidate variables
  unsigned int n_pfcand_;
  float  pfcan_pt_[100];
  float  pfcan_phirel_[100];
  float  pfcan_etarel_[100];
  float  pfcan_puppiw_[100];


};


DeepNtuplizer::DeepNtuplizer(const edm::ParameterSet& iConfig):
  vtxToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
  jetToken_(consumes<pat::JetCollection>(iConfig.getParameter<edm::InputTag>("jets")))
{
  //now do what ever initialization is needed
  usesResource("TFileService");

  tree_->Branch("gen_pt"    ,&gen_pt_    ,"gen_pt_/f"    );
  // global even variables
  tree_->Branch("npv"    ,&npv_    ,"npv_/i"    );

  // jet variables
  tree_->Branch("jet_pt", &jet_pt_);
  tree_->Branch("jet_eta", &jet_eta_);

  // PFcanditates per jet
  tree_->Branch("n_pfcand", &n_pfcand_,"n_pfcand_/i");
  tree_->Branch("nPF_pt", &pfcan_pt_,"pfcan_pt_[n_pfcand_]/f");
  tree_->Branch("pfcan_phirel",&pfcan_phirel_,"pfcan_phirel_[n_pfcand_]/f");
  tree_->Branch("pfcan_etarel",&pfcan_etarel_,"pfcan_etarel_[n_pfcand_]/f");
  tree_->Branch("pfcan_puppiw",&pfcan_puppiw_,"pfcan_puppiw_[n_pfcand_]/f");

}


DeepNtuplizer::~DeepNtuplizer()
{
  file_->Close();
}


// ------------ method called for each event  ------------
void
DeepNtuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);
  if (vertices->empty()) return; // skip the event if no PV found
  const reco::Vertex &PV = vertices->front();

  edm::Handle<pat::JetCollection> jets;
  iEvent.getByToken(jetToken_, jets);

  // clear vectors
  npv_ = vertices->size();

  const static float magic_invalid = -999;

  // loop over the jets
  for (const pat::Jet &jet : *jets) {

    // label: this is what we want to know in data, but only have in MC
    gen_pt_ = magic_invalid;
    if(jet.genJet()!=NULL)   gen_pt_ =  jet.genJet()->pt();


    // global jet quanties
      jet_pt_ = jet.pt();
      jet_eta_ = jet.eta();
      float etasign = 1.;
      if (jet.eta()<0) etasign =-1.;

      n_pfcand_ =  jet.numberOfDaughters();
      for (unsigned int i = 0; i <  jet.numberOfDaughters(); i++)
	{
	  const pat::PackedCandidate* PackedCandidate_ = dynamic_cast<const pat::PackedCandidate*>(jet.daughter(i));
	  pfcan_pt_[i] = PackedCandidate_->pt();
	  pfcan_phirel_[i] = reco::deltaPhi(PackedCandidate_->phi(),jet.phi());
	  pfcan_etarel_[i] = etasign*(PackedCandidate_->eta()-jet.eta());
	  pfcan_puppiw_[i] = PackedCandidate_->puppiWeight();	  
	}  
 



      tree_->Fill();
  }
}


// ------------ method called once each job just before starting event loop  ------------
void
DeepNtuplizer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
DeepNtuplizer::endJob()
{
  file_->cd();
  tree_->Write();
  file_->Write();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DeepNtuplizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepNtuplizer);
