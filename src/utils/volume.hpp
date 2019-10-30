#pragma once

#include <fstream>
#include <string>
#include <memory>

#include <cudafx/misc.hpp>
#include <cudafx/memory.hpp>
#include <VMUtils/concepts.hpp>
#include <VMUtils/modules.hpp>
#include <vocomp/index/index.hpp>
#include <vocomp/refine/extractor.hpp>
#include <vocomp/refine/pipe_factory.hpp>
#include <vocomp/video/decompressor.hpp>

VM_BEGIN_MODULE( vol )

using namespace std;
namespace cuda = cufx;

template <typename Voxel>
struct VolumeInner : vm::NoCopy, vm::NoMove
{
	VolumeInner( std::string const &file_name ) :
	  file_name( file_name ),
	  is( [&] {
		  std::ifstream is( file_name, ios::binary | ios::ate );
		  if ( !is ) {
			  throw std::runtime_error( vm::fmt( "unable to open input file: {}", file_name ) );
		  }
		  return is;
	  }() ),
	  reader( is, 0, is.tellg() ),
	  _( reader ),
	  block_dim( cuda::Extent{}
				   .set_width( _.block_size() )
				   .set_height( _.block_size() )
				   .set_depth( _.block_size() ) ),
	  block_size( sizeof( Voxel ) * block_dim.size() ),
	  decomp( PipeFactory::create( file_name ) )
	{
	}

	string file_name;
	ifstream is;
	StreamReader reader;
	vol::refine::Extractor _;
	cuda::Extent block_dim;
	size_t block_size;
	std::shared_ptr<vol::Pipe> decomp;
};

VM_EXPORT
{
	template <typename Voxel>
	struct Volume;

	template <typename Voxel>
	struct ArchivedVolumeBlock;

	template <typename Voxel>
	struct VolumeBlock
	{
	private:
		struct Inner : vm::NoCopy, vm::NoMove
		{
			string _;
			cuda::Extent dim;
		};

	public:
		const Voxel *data() const
		{
			return reinterpret_cast<const Voxel *>( _->_.c_str() );
		}
		cuda::MemoryView3D<Voxel> view() const
		{
			auto view_info = cuda::MemoryView2DInfo{}
							   .set_stride( this->_->dim.width * sizeof( Voxel ) )
							   .set_width( this->_->dim.width )
							   .set_height( this->_->dim.height );
			return cuda::MemoryView3D<Voxel>( const_cast<char *>( _->_.c_str() ),
											  view_info, this->_->dim );
		}

	private:
		VolumeBlock( string &&_, cuda::Extent dim )
		{
			this->_->_ = std::move( _ );
			this->_->dim = dim;
		}

		shared_ptr<Inner> _ = make_shared<Inner>();
		friend struct ArchivedVolumeBlock<Voxel>;
	};

	template <typename Voxel>
	struct ArchivedVolumeBlock
	{
		VolumeBlock<Voxel> unarchive() const
		{
			string buffer;
			buffer.resize( _->block_size );
			auto buffer_ptr = const_cast<char *>( buffer.c_str() );
			SliceWriter writer( buffer_ptr, _->block_size );
			auto i = index::Idx{}
					   .set_x( idx.x )
					   .set_y( idx.y )
					   .set_z( idx.z );
			auto reader = _->_.extract( i );
			_->decomp->transfer( reader, writer );
			return VolumeBlock<Voxel>( std::move( buffer ), _->block_dim );
		}
		cufx::Extent unarchive_into( cufx::MemoryView1D<unsigned char> const &swap ) const
		{
			auto i = index::Idx{}
					   .set_x( idx.x )
					   .set_y( idx.y )
					   .set_z( idx.z );
			auto reader = _->_.extract( i );
			if ( auto video = dynamic_cast<vol::video::Decompressor *>( _->decomp.get() ) ) {
				auto &dim = _->block_dim;
				video->decompress( reader, swap );
				return dim;
			} else {
				throw std::runtime_error( "unsupported archive" );
			}
		}
		uint3 index() const { return idx; }

	private:
		shared_ptr<VolumeInner<Voxel>> _;
		size_t offset;
		uint3 idx;
		friend struct Volume<Voxel>;
	};

	template <typename Voxel>
	struct Volume
	{
		static Volume from_compressed( const string &file_name )
		{
			Volume vol;
			vol._ = make_shared<VolumeInner<Voxel>>( file_name );
			auto len = vol._->_.block_size();
			vol.block_stride = vol._->block_size;
			vol.grid_dim = dim3( unsigned( vol._->_.adjusted().x / len ),
								 unsigned( vol._->_.adjusted().y / len ),
								 unsigned( vol._->_.adjusted().z / len ) );
			vol.all = vol.grid_dim * ( len - vol._->_.padding() * 2 );
			vol.raw_all = dim3{ unsigned( vol._->_.raw().x ),
								unsigned( vol._->_.raw().y ),
								unsigned( vol._->_.raw().z ) };
			return vol;
		}

	public:
		dim3 dim() const { return grid_dim; }
		cuda::Extent block_dim() const { return _->block_dim; }
		uint3 padding() const { return uint3{ unsigned( _->_.padding() ),
											  unsigned( _->_.padding() ),
											  unsigned( _->_.padding() ) }; }

		ArchivedVolumeBlock<Voxel> get_block( uint3 idx )
		{
			auto block_id = idx.x +
							idx.y * grid_dim.x +
							idx.z * grid_dim.x * grid_dim.y;
			ArchivedVolumeBlock<Voxel> block;
			block._ = _;
			block.offset = offset + block_id * block_stride + block_offset;
			block.idx = idx;
			return block;
		}
		vector<ArchivedVolumeBlock<Voxel>> get_blocks( uint3 idmin, uint3 idmax )
		{
			uint3 idx;
			vector<ArchivedVolumeBlock<Voxel>> _;
			for ( idx.z = idmin.z; idx.z < idmax.z; ++idx.z ) {
				for ( idx.y = idmin.y; idx.y < idmax.y; ++idx.y ) {
					for ( idx.x = idmin.x; idx.x < idmax.x; ++idx.x ) {
						_.emplace_back( get_block( idx ) );
					}
				}
			}
			return _;
		}
		vector<ArchivedVolumeBlock<Voxel>> get_blocks()
		{
			return get_blocks( uint3{ 0, 0, 0 }, uint3{ grid_dim.x, grid_dim.y, grid_dim.z } );
		}
		float scale() const
		{
			return min( min( float( all.x ) / raw_all.x,
							 float( all.y ) / raw_all.y ),
						float( all.z ) / raw_all.z );
		}

	private:
		shared_ptr<VolumeInner<Voxel>> _;
		dim3 grid_dim = dim3( 1, 1, 1 );
		size_t offset = 0;
		size_t block_stride = 0;
		size_t block_offset = 0;
		dim3 all, raw_all;
	};
}

VM_END_MODULE()
